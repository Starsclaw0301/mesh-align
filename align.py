"""
mesh-align: 将高精度网格模型对齐到低精度网格模型的坐标系
支持格式：STL / OBJ（目标和源均可）
方案：FPFH 特征提取 + RANSAC 全局配准 + ICP 精配准
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

def load_and_sample(path: str, n_points: int = 50000) -> o3d.geometry.PointCloud:
    """加载 STL/OBJ 并均匀采样点云"""
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    points, _ = trimesh.sample.sample_surface(mesh, n_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """下采样 + 法线估算 + FPFH 特征"""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    down.orient_normals_consistent_tangent_plane(10)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return down, fpfh


# ─────────────────────────────────────────
# 2. RANSAC 全局配准
# ─────────────────────────────────────────

def ransac_registration(
    src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999),
    )
    return result


# ─────────────────────────────────────────
# 3. ICP 精配准（两阶段）
# ─────────────────────────────────────────

def refine_icp(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """point-to-plane ICP 两阶段精配准"""
    icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    # 阶段1：中等精度
    r1 = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=voxel_size * 0.4,
        init=init_transform,
        estimation_method=icp_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200
        ),
    )

    # 阶段2：收紧到 0.1mm
    r2 = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=0.0001,
        init=r1.transformation,
        estimation_method=icp_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=500
        ),
    )

    return r2 if r2.fitness > r1.fitness * 0.5 else r1


# ─────────────────────────────────────────
# 4. 精度验证
# ─────────────────────────────────────────

def evaluate(src_pcd, tgt_pcd, transform) -> dict:
    src_t = o3d.geometry.PointCloud(src_pcd)
    src_t.transform(transform)
    dists = np.asarray(src_t.compute_point_cloud_distance(tgt_pcd))
    return {
        "mean_mm":   dists.mean() * 1000,
        "median_mm": np.median(dists) * 1000,
        "p95_mm":    np.percentile(dists, 95) * 1000,
        "max_mm":    dists.max() * 1000,
    }


# ─────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="将高精度网格对齐到低精度网格的坐标系（FPFH+RANSAC+ICP）")
    parser.add_argument("target", help="目标坐标系网格文件（低精度，STL 或 OBJ）")
    parser.add_argument("source", help="待对齐网格文件（高精度，STL 或 OBJ）")
    parser.add_argument("-o", "--output", default="aligned.obj", help="输出文件路径（默认 aligned.obj）")
    parser.add_argument("-n", "--n-points", type=int, default=50000, help="采样点数（默认 50000）")
    parser.add_argument("-v", "--voxel-size", type=float, default=0.0, help="体素大小（米），0 表示自动推算")
    parser.add_argument("--save-matrix", default="transform.npy", help="保存 4×4 变换矩阵路径")
    args = parser.parse_args()

    print(f"[1/5] 加载目标网格: {args.target}")
    tgt_pcd = load_and_sample(args.target, args.n_points)

    print(f"[2/5] 加载源网格: {args.source}")
    src_pcd = load_and_sample(args.source, args.n_points)

    # 自动推算 voxel_size：模型对角线的 1%
    if args.voxel_size <= 0:
        pts = np.asarray(tgt_pcd.points)
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        voxel_size = diag * 0.01
    else:
        voxel_size = args.voxel_size
    print(f"      voxel_size = {voxel_size*1000:.2f} mm")

    print("[3/5] 提取 FPFH 特征...")
    src_down, src_fpfh = preprocess(src_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess(tgt_pcd, voxel_size)
    print(f"      下采样点数: src={len(src_down.points)}  tgt={len(tgt_down.points)}")

    print("[4/5] RANSAC 全局配准...")
    ransac_result = ransac_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size)
    print(f"      RANSAC → RMSE={ransac_result.inlier_rmse*1000:.4f}mm  fitness={ransac_result.fitness:.4f}")

    print("      ICP 精配准...")
    icp_result = refine_icp(src_pcd, tgt_pcd, ransac_result.transformation, voxel_size)
    print(f"      ICP   → RMSE={icp_result.inlier_rmse*1000:.4f}mm  fitness={icp_result.fitness:.4f}")

    best_T = icp_result.transformation

    # 精度验证
    print("\n[精度验证] 最近邻点距离统计...")
    stats = evaluate(src_pcd, tgt_pcd, best_T)
    print(f"  Mean:   {stats['mean_mm']:.4f} mm")
    print(f"  Median: {stats['median_mm']:.4f} mm")
    print(f"  P95:    {stats['p95_mm']:.4f} mm")
    print(f"  Max:    {stats['max_mm']:.4f} mm")

    ok = stats['median_mm'] <= 0.1
    print(f"\n{'✅' if ok else '⚠️ '} 中位误差 {stats['median_mm']:.4f}mm {'达到' if ok else '未达到'} 0.1mm 目标")

    print(f"\n变换矩阵 (4×4):\n{np.round(best_T, 8)}")
    np.save(args.save_matrix, best_T)
    print(f"\n[5/5] 变换矩阵已保存: {args.save_matrix}")

    # 保留材质导出
    mesh_src = trimesh.load(args.source)
    mesh_src.apply_transform(best_T)
    mesh_src.export(args.output)
    print(f"对齐后的网格已保存: {args.output}")


if __name__ == "__main__":
    main()
