"""
mesh-align: 将高精度 OBJ 模型对齐到低精度 STL 模型的坐标系
方案：PCA 粗配准 + ICP 精配准
目标精度：0.1mm (0.0001m)
"""

import argparse
import sys
import numpy as np
import open3d as o3d
import trimesh

# ─────────────────────────────────────────
# 1. 加载网格并采样点云
# ─────────────────────────────────────────

def load_and_sample(path: str, n_points: int = 50000) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """加载 STL/OBJ 并均匀采样点云，返回 (PointCloud, vertices_array)"""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    # 均匀面积采样，保证低精度模型也有足够采样点
    points, _ = trimesh.sample.sample_surface_even(mesh, n_points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(pcd, 10)
    
    return pcd, np.asarray(mesh.vertices)


# ─────────────────────────────────────────
# 2. PCA 粗配准
# ─────────────────────────────────────────

def pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (centroid, 3×3 主轴矩阵，列为主轴方向，按特征值降序)"""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvecs[:, order]


def pca_rough_align(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
) -> list[np.ndarray]:
    """
    用 PCA 生成候选变换矩阵列表。
    对称歧义：每个主轴可能差 180°，共 4 种翻转组合（保持右手系）。
    返回 4 个 4×4 变换矩阵。
    """
    src_pts = np.asarray(src_pcd.points)
    tgt_pts = np.asarray(tgt_pcd.points)

    src_c, src_axes = pca_axes(src_pts)
    tgt_c, tgt_axes = pca_axes(tgt_pts)

    # src_axes 的列向量：src 坐标系中的主轴
    # 目标：把 src 的主轴旋转到与 tgt 主轴对齐
    # R = tgt_axes @ src_axes^T

    candidates = []
    # 4 种符号翻转：(ax0, ax1) 各 ±1，ax2 由右手系决定
    for s0 in [1, -1]:
        for s1 in [1, -1]:
            sa = src_axes.copy()
            sa[:, 0] *= s0
            sa[:, 1] *= s1
            sa[:, 2] = np.cross(sa[:, 0], sa[:, 1])  # 保持右手系

            R = tgt_axes @ sa.T  # 3×3

            # 构造 4×4：先平移 src 到原点，旋转，再平移到 tgt 中心
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tgt_c - R @ src_c
            candidates.append(T)

    return candidates


# ─────────────────────────────────────────
# 3. ICP 精配准
# ─────────────────────────────────────────

def run_icp(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_dist: float = 0.005,   # 初始最大对应距离 5mm
    fine_dist: float = 0.0001, # 精配准收紧到 0.1mm
) -> o3d.pipelines.registration.RegistrationResult:
    """两阶段 ICP：先粗后精"""

    # 粗 ICP
    result_coarse = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=max_dist,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=200
        ),
    )

    # 精 ICP（从粗配准结果继续）
    result_fine = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=fine_dist,
        init=result_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-9, relative_rmse=1e-9, max_iteration=500
        ),
    )

    return result_fine


# ─────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="将高精度 OBJ 对齐到低精度 STL 坐标系")
    parser.add_argument("stl", help="低精度 STL 文件路径（目标坐标系）")
    parser.add_argument("obj", help="高精度 OBJ 文件路径（需对齐）")
    parser.add_argument("-o", "--output", default="aligned.obj", help="输出 OBJ 路径")
    parser.add_argument("-n", "--n-points", type=int, default=50000, help="采样点数（默认 50000）")
    parser.add_argument("--save-matrix", default="transform.npy", help="保存 4×4 变换矩阵路径")
    args = parser.parse_args()

    print(f"[1/5] 加载 STL: {args.stl}")
    tgt_pcd, _ = load_and_sample(args.stl, args.n_points)

    print(f"[2/5] 加载 OBJ: {args.obj}")
    src_pcd, _ = load_and_sample(args.obj, args.n_points)

    print("[3/5] PCA 粗配准（生成 4 个候选变换）...")
    candidates = pca_rough_align(src_pcd, tgt_pcd)

    print("[4/5] ICP 精配准（逐候选测试，取最优）...")
    best_result = None
    best_rmse = float("inf")

    for i, T_init in enumerate(candidates):
        try:
            result = run_icp(src_pcd, tgt_pcd, T_init)
            rmse_mm = result.inlier_rmse * 1000
            fitness = result.fitness
            print(f"  候选 {i+1}/4 → RMSE={rmse_mm:.4f}mm  fitness={fitness:.4f}")
            if result.inlier_rmse < best_rmse and fitness > 0.1:
                best_rmse = result.inlier_rmse
                best_result = result
        except Exception as e:
            print(f"  候选 {i+1} 失败: {e}")

    if best_result is None:
        print("❌ 所有候选配准均失败，请检查输入模型。")
        sys.exit(1)

    best_T = best_result.transformation
    best_rmse_mm = best_result.inlier_rmse * 1000
    print(f"\n✅ 最优配准 RMSE = {best_rmse_mm:.4f} mm  (目标 ≤ 0.1mm)")
    if best_rmse_mm > 0.1:
        print(f"⚠️  RMSE 超过目标精度，建议增大采样点数 (-n) 或检查模型一致性。")

    print(f"\n变换矩阵 (4×4):\n{np.round(best_T, 8)}")

    # 保存变换矩阵
    np.save(args.save_matrix, best_T)
    print(f"\n[5/5] 变换矩阵已保存: {args.save_matrix}")

    # 应用变换并输出对齐后的 OBJ
    mesh_src = trimesh.load(args.obj, force="mesh")
    if isinstance(mesh_src, trimesh.Scene):
        mesh_src = trimesh.util.concatenate(list(mesh_src.geometry.values()))
    mesh_src.apply_transform(best_T)
    mesh_src.export(args.output)
    print(f"对齐后的 OBJ 已保存: {args.output}")


if __name__ == "__main__":
    main()
