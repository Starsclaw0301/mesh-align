#!/usr/bin/env python3
"""将高精度 OBJ 模型对齐到低精度 STL/OBJ 参考模型的坐标系。

原理:
  1. 从两个网格分别采样点云
  2. 用 FPFH 特征 + RANSAC 做全局粗配准 (解决大角度偏移)
  3. 用 ICP 做精细配准 (亚毫米级精度)
  4. 将变换矩阵应用到高精度 OBJ (保留材质/UV/法线), 输出对齐后的 OBJ

依赖:  pip install open3d trimesh numpy
用法:
  # 单对对齐
  python align_meshes.py \
      --ref  chairbot-urdf-0212/meshes/base_link.STL \
      --src  id_mesh/meshes_aligned/base_link.obj \
      --out  id_mesh/meshes_output/base_link.obj

  # 批量对齐 (自动匹配同名零件)
  python align_meshes.py \
      --ref-dir  chairbot-urdf-0212/meshes \
      --src-dir  id_mesh/meshes_aligned \
      --out-dir  id_mesh/meshes_output \
      --ref-ext .STL --src-ext .obj

  # 可选参数
  --voxel-size 0.005   # 点云降采样体素大小 (米), 影响速度和精度
  --max-icp-dist 0.01  # ICP 最大对应点距离 (米)
  --no-global           # 跳过全局配准 (若初始对齐已大致正确)
  --no-scale            # 禁用自动尺度补偿 (默认自动检测 10x/100x 等尺度差异)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("需要 open3d: pip install open3d", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
except ImportError:
    print("需要 trimesh: pip install trimesh", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────
# 1. 网格 → 点云
# ──────────────────────────────────────────────

def mesh_to_pointcloud(mesh_path: str | Path, num_points: int = 50000) -> o3d.geometry.PointCloud:
    """加载 STL/OBJ 网格并采样为 Open3D 点云。"""
    mesh_path = Path(mesh_path)
    # 用 trimesh 加载 (兼容 STL/OBJ/PLY 等)
    tm = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(tm, trimesh.Scene):
        meshes = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
        tm = trimesh.util.concatenate(meshes)

    # 转换为 Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    o3d_mesh.compute_vertex_normals()

    # 均匀采样点云
    pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd



# ──────────────────────────────────────────────
# 2. 尺度估计与自动参数
# ──────────────────────────────────────────────

def _get_pcd_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    """计算点云 AABB 对角线长度。"""
    bb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))


def estimate_scale(pcd_ref: o3d.geometry.PointCloud,
                   pcd_src: o3d.geometry.PointCloud) -> float:
    """通过 AABB 对角线比值估算两个点云的尺度比, 并 snap 到常见整数倍。

    返回 scale 使得 src * scale ≈ ref 的尺度。

    当 raw ratio 接近 10 的整数次幂 (0.001, 0.01, 0.1, 1, 10, 100, 1000)
    时, 自动 snap 到该整数倍, 避免浮点漂移。
    """
    ref_diag = _get_pcd_diagonal(pcd_ref)
    src_diag = _get_pcd_diagonal(pcd_src)
    if src_diag < 1e-12:
        return 1.0

    raw_ratio = ref_diag / src_diag

    # 常见尺度因子 (覆盖 mm↔m, cm↔m, inch↔mm 等)
    NICE_FACTORS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    SNAP_TOLERANCE = 0.30  # 30% 容差

    best_factor = raw_ratio
    best_err = float("inf")
    for f in NICE_FACTORS:
        rel_err = abs(raw_ratio - f) / f
        if rel_err < best_err:
            best_err = rel_err
            best_factor = f

    if best_err < SNAP_TOLERANCE:
        print(f"  尺度检测: raw_ratio={raw_ratio:.4f} → snap 到 {best_factor}")
        return float(best_factor)
    else:
        print(f"  尺度检测: raw_ratio={raw_ratio:.4f} (未匹配到整数倍, 直接使用)")
        return float(raw_ratio)


def auto_params(pcd_ref: o3d.geometry.PointCloud,
                pcd_src: o3d.geometry.PointCloud
                ) -> Tuple[float, float]:
    """根据模型实际尺寸自动计算 voxel_size 和 max_icp_dist。

    策略:
      voxel_size   = 对角线的 1.5%
      max_icp_dist = 对角线的 1%   (之前 5% 太大，容易错误匹配)
    """
    diag = max(_get_pcd_diagonal(pcd_ref), _get_pcd_diagonal(pcd_src), 1e-6)
    voxel_size = diag * 0.015
    max_icp_dist = diag * 0.01
    return voxel_size, max_icp_dist


# ──────────────────────────────────────────────
# 3. FPFH 特征提取
# ──────────────────────────────────────────────

def preprocess_pcd(pcd: o3d.geometry.PointCloud,
                   voxel_size: float
                   ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """降采样 + 法线估计 + FPFH 特征计算。"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


# ──────────────────────────────────────────────
# 4. 全局配准 (RANSAC + FPFH)
# ──────────────────────────────────────────────

def global_registration(
    src_down: o3d.geometry.PointCloud,
    ref_down: o3d.geometry.PointCloud,
    src_fpfh: o3d.pipelines.registration.Feature,
    ref_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """RANSAC 全局配准: 通过 FPFH 特征匹配找到粗略位姿。"""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=src_down,
        target=ref_down,
        source_feature=src_fpfh,
        target_feature=ref_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.9999),
    )
    return result


# ──────────────────────────────────────────────
# 5. ICP 精细配准
# ──────────────────────────────────────────────

def refine_icp(
    src_pcd: o3d.geometry.PointCloud,
    ref_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_distance: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """Point-to-Plane ICP 多阶段精细配准，最终收紧到 0.1mm。"""
    icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    def _ensure_normals(pcd, radius):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
        )
        o3d.geometry.PointCloud.orient_normals_consistent_tangent_plane(pcd, 20)

    # 阶段1~3：从粗到细，基于 max_distance 逐步收紧
    current_transform = init_transform
    for dist_factor, max_iter in [(5.0, 100), (2.0, 200), (1.0, 300)]:
        dist = max_distance * dist_factor
        _ensure_normals(src_pcd, dist * 2)
        _ensure_normals(ref_pcd, dist * 2)
        r = o3d.pipelines.registration.registration_icp(
            source=src_pcd,
            target=ref_pcd,
            max_correspondence_distance=dist,
            init=current_transform,
            estimation_method=icp_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=max_iter,
            ),
        )
        current_transform = r.transformation

    # 阶段4：固定收紧到 1mm
    _ensure_normals(src_pcd, 0.002)
    _ensure_normals(ref_pcd, 0.002)
    r = o3d.pipelines.registration.registration_icp(
        source=src_pcd, target=ref_pcd,
        max_correspondence_distance=0.001,
        init=current_transform,
        estimation_method=icp_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=500,
        ),
    )
    current_transform = r.transformation

    # 阶段5：最终收紧到 0.1mm（目标精度）
    _ensure_normals(src_pcd, 0.0005)
    _ensure_normals(ref_pcd, 0.0005)
    r_final = o3d.pipelines.registration.registration_icp(
        source=src_pcd, target=ref_pcd,
        max_correspondence_distance=0.0001,
        init=current_transform,
        estimation_method=icp_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=1000,
        ),
    )

    # 若最终阶段 fitness 太低（说明对齐还不够好，0.1mm 容差找不到足够对应点），回退到上一阶段
    return r_final if r_final.fitness > r.fitness * 0.5 else r


# ──────────────────────────────────────────────
# 6. OBJ 文本级变换 (保留材质/UV)
# ──────────────────────────────────────────────

def apply_transform_to_obj(
    src_obj_path: Path,
    out_obj_path: Path,
    transform_4x4: np.ndarray,
    scale: float = 1.0,
):
    """在文本级别对 OBJ 施加刚体变换, 完美保留 UV/材质/法线。

    transform_4x4: 4x4 齐次变换矩阵 (src → ref 坐标系)
    scale: 若有尺度补偿, 先缩放再变换
    """
    rot = transform_4x4[:3, :3]
    trans = transform_4x4[:3, 3]

    lines = src_obj_path.read_text(errors="ignore").splitlines()
    out_lines = []

    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            pt = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * scale
            pt_new = rot @ pt + trans
            out_lines.append(f"v {pt_new[0]:.8f} {pt_new[1]:.8f} {pt_new[2]:.8f}")
        elif line.startswith("vn "):
            parts = line.split()
            n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            n_new = rot @ n
            length = np.linalg.norm(n_new)
            if length > 1e-12:
                n_new /= length
            out_lines.append(f"vn {n_new[0]:.8f} {n_new[1]:.8f} {n_new[2]:.8f}")
        elif line.startswith("mtllib "):
            # 更新 mtllib 引用为输出文件名
            out_lines.append(f"mtllib {out_obj_path.stem}.mtl")
        else:
            out_lines.append(line)

    out_obj_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj_path.write_text("\n".join(out_lines) + "\n")

    # 复制 mtl 文件
    src_mtl = src_obj_path.with_suffix(".mtl")
    out_mtl = out_obj_path.with_suffix(".mtl")
    if src_mtl.exists():
        import shutil
        try:
            # 若目标已存在且只读, 先删除
            if out_mtl.exists():
                out_mtl.chmod(0o644)
            shutil.copyfile(str(src_mtl), str(out_mtl))
        except (PermissionError, OSError):
            try:
                out_mtl.write_bytes(src_mtl.read_bytes())
            except (PermissionError, OSError) as exc:
                print(f"    [WARN] mtl 复制失败: {exc}", file=sys.stderr)

    return out_obj_path


# ──────────────────────────────────────────────
# 7. 单对对齐核心流程
# ──────────────────────────────────────────────

def align_one_pair(
    ref_path: Path,
    src_path: Path,
    out_path: Path,
    voxel_size: float = 0.005,
    max_icp_dist: float = 0.01,
    num_points: int = 50000,
    do_global: bool = True,
    no_scale: bool = False,
) -> dict:
    """对齐单对模型。

    Args:
        ref_path:  低精度参考网格 (STL/OBJ)
        src_path:  高精度待对齐网格 (OBJ)
        out_path:  输出对齐后 OBJ 路径
        voxel_size: 点云降采样体素大小
        max_icp_dist: ICP 最大对应点距离
        num_points: 采样点数
        do_global: 是否做全局粗配准
        no_scale: 禁用自动尺度补偿 (默认自动检测并补偿)

    Returns:
        dict 包含 transform, scale, fitness, rmse 等信息
    """
    print(f"  加载参考网格: {ref_path.name}")
    pcd_ref = mesh_to_pointcloud(ref_path, num_points)

    print(f"  加载源网格:   {src_path.name}")
    pcd_src = mesh_to_pointcloud(src_path, num_points)

    # ---- 尺度自动检测与补偿 (默认开启) ----
    scale = 1.0
    ref_diag = _get_pcd_diagonal(pcd_ref)
    src_diag = _get_pcd_diagonal(pcd_src)
    print(f"  ref 对角线={ref_diag:.4f}, src 对角线={src_diag:.4f}")

    if not no_scale:
        scale = estimate_scale(pcd_ref, pcd_src)
        if abs(scale - 1.0) > 0.01:
            print(f"  应用尺度补偿: src × {scale}")
            pts = np.asarray(pcd_src.points) * scale
            pcd_src.points = o3d.utility.Vector3dVector(pts)
        else:
            scale = 1.0
            print(f"  尺度接近 1.0, 无需补偿")
    else:
        print(f"  尺度补偿已禁用 (--no-scale)")

    # ---- 自动参数估算 (在尺度补偿之后计算, 确保参数正确) ----
    if voxel_size <= 0 or max_icp_dist <= 0:
        auto_vs, auto_icp = auto_params(pcd_ref, pcd_src)
        if voxel_size <= 0:
            voxel_size = auto_vs
        if max_icp_dist <= 0:
            max_icp_dist = auto_icp
        print(f"  自动参数: voxel_size={voxel_size:.4f}, max_icp_dist={max_icp_dist:.4f}")

    # ---- 全局配准 ----
    init_transform = np.eye(4)
    if do_global:
        print(f"  FPFH 特征提取 (voxel_size={voxel_size:.4f}) ...")
        src_down, src_fpfh = preprocess_pcd(pcd_src, voxel_size)
        ref_down, ref_fpfh = preprocess_pcd(pcd_ref, voxel_size)

        print(f"  RANSAC 全局粗配准 ...")
        global_result = global_registration(src_down, ref_down, src_fpfh, ref_fpfh, voxel_size)
        init_transform = global_result.transformation
        print(f"    粗配准 fitness={global_result.fitness:.4f}, RMSE={global_result.inlier_rmse:.6f}")

    # ---- ICP 精细配准 ----
    print(f"  ICP 精细配准 ...")
    # 起始距离取 RANSAC RMSE 的 3 倍（保证初始有足够对应点），再逐步收紧
    if do_global:
        icp_start_dist = max(global_result.inlier_rmse * 3, max_icp_dist)
    else:
        icp_start_dist = max_icp_dist
    print(f"    ICP 起始距离={icp_start_dist*1000:.2f}mm")
    icp_result = refine_icp(pcd_src, pcd_ref, init_transform, icp_start_dist)

    final_transform = icp_result.transformation
    print(f"    精配准 fitness={icp_result.fitness:.4f}, RMSE={icp_result.inlier_rmse*1000:.4f}mm")

    # 精度验证：最近邻点距离统计
    src_eval = o3d.geometry.PointCloud(pcd_src)
    src_eval.transform(final_transform)
    dists = np.asarray(src_eval.compute_point_cloud_distance(pcd_ref)) * 1000  # mm
    print(f"    [精度验证] mean={dists.mean():.4f}mm  median={np.median(dists):.4f}mm  "
          f"p95={np.percentile(dists,95):.4f}mm  max={dists.max():.4f}mm")
    if np.median(dists) <= 0.1:
        print(f"    ✅ 中位误差 {np.median(dists):.4f}mm ≤ 0.1mm，达到目标精度")
    else:
        print(f"    ⚠️  中位误差 {np.median(dists):.4f}mm > 0.1mm，未达到目标精度")

    # ---- 应用变换到 OBJ ----
    print(f"  输出对齐后 OBJ: {out_path}")
    if src_path.suffix.lower() == ".obj":
        apply_transform_to_obj(src_path, out_path, final_transform, scale)
    else:
        # STL 等格式: 用 trimesh 变换后导出
        tm = trimesh.load(str(src_path), force="mesh")
        if scale != 1.0:
            tm.vertices *= scale
        tm.apply_transform(final_transform)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(str(out_path))

    return {
        "ref": str(ref_path),
        "src": str(src_path),
        "out": str(out_path),
        "transform": final_transform.tolist(),
        "scale": scale,
        "fitness": icp_result.fitness,
        "rmse": icp_result.inlier_rmse,
    }


# ──────────────────────────────────────────────
# 8. 批量对齐
# ──────────────────────────────────────────────

def batch_align(
    ref_dir: Path,
    src_dir: Path,
    out_dir: Path,
    ref_ext: str = ".STL",
    src_ext: str = ".obj",
    **kwargs,
) -> list[dict]:
    """批量对齐同名零件。"""
    ref_files = sorted(ref_dir.glob(f"*{ref_ext}"))
    results = []
    success = 0
    fail = 0

    for ref_file in ref_files:
        part_name = ref_file.stem
        src_file = src_dir / f"{part_name}{src_ext}"
        if not src_file.exists():
            # 尝试大小写不敏感匹配
            candidates = [f for f in src_dir.iterdir()
                          if f.stem.lower() == part_name.lower() and f.suffix.lower() == src_ext.lower()]
            if candidates:
                src_file = candidates[0]
            else:
                print(f"  [跳过] {part_name}: 未找到对应的源模型 {src_ext}")
                continue

        out_file = out_dir / f"{part_name}.obj"
        print(f"\n{'='*60}")
        print(f"对齐: {part_name}")
        print(f"{'='*60}")

        try:
            result = align_one_pair(ref_file, src_file, out_file, **kwargs)
            results.append(result)
            success += 1
            print(f"  ✓ fitness={result['fitness']:.4f}, RMSE={result['rmse']:.6f}")
        except Exception as exc:
            print(f"  ✗ 失败: {exc}", file=sys.stderr)
            fail += 1
            results.append({"ref": str(ref_file), "src": str(src_file), "error": str(exc)})

    print(f"\n{'='*60}")
    print(f"批量对齐完成: 成功 {success}/{success+fail}, 失败 {fail}")
    print(f"{'='*60}")

    # 保存变换矩阵记录
    import json
    record_path = out_dir / "alignment_record.json"
    record_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"变换记录已保存: {record_path}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将高精度模型对齐到低精度参考模型的坐标系 (FPFH + RANSAC + ICP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单对对齐
  python align_meshes.py --ref ref.STL --src high.obj --out aligned.obj

  # 批量对齐 (自动匹配同名零件)
  python align_meshes.py \\
      --ref-dir chairbot-urdf-0212/meshes \\
      --src-dir id_mesh/meshes_aligned \\
      --out-dir id_mesh/meshes_output

  # 禁用自动尺度补偿 (若确定两模型尺度一致)
  python align_meshes.py --ref ref.STL --src high.obj --out aligned.obj --no-scale
        """,
    )

    # 单对模式
    parser.add_argument("--ref", type=Path, help="低精度参考网格路径 (STL/OBJ)")
    parser.add_argument("--src", type=Path, help="高精度待对齐网格路径 (OBJ)")
    parser.add_argument("--out", type=Path, help="输出对齐后网格路径")

    # 批量模式
    parser.add_argument("--ref-dir", type=Path, help="低精度参考网格目录")
    parser.add_argument("--src-dir", type=Path, help="高精度待对齐网格目录")
    parser.add_argument("--out-dir", type=Path, help="输出目录")
    parser.add_argument("--ref-ext", default=".STL", help="参考模型扩展名 (默认 .STL)")
    parser.add_argument("--src-ext", default=".obj", help="源模型扩展名 (默认 .obj)")

    # 算法参数
    parser.add_argument("--voxel-size", type=float, default=0,
                        help="降采样体素大小 (默认 0=自动, 按模型对角线 1.5%% 估算)")
    parser.add_argument("--max-icp-dist", type=float, default=0,
                        help="ICP 最大对应点距离 (默认 0=自动, 按模型对角线 5%% 估算)")
    parser.add_argument("--num-points", type=int, default=100000,
                        help="采样点数 (默认 100000)")
    parser.add_argument("--no-global", action="store_true",
                        help="跳过全局配准 (仅当初始对齐已大致正确时)")
    parser.add_argument("--no-scale", action="store_true",
                        help="禁用自动尺度补偿 (默认自动检测 mm/cm/m 等尺度差异)")

    args = parser.parse_args()

    common_kwargs = dict(
        voxel_size=args.voxel_size,
        max_icp_dist=args.max_icp_dist,
        num_points=args.num_points,
        do_global=not args.no_global,
        no_scale=args.no_scale,
    )

    if args.ref_dir and args.src_dir:
        # 批量模式
        out_dir = args.out_dir or (args.src_dir.parent / "meshes_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        batch_align(
            args.ref_dir, args.src_dir, out_dir,
            ref_ext=args.ref_ext, src_ext=args.src_ext,
            **common_kwargs,
        )
    elif args.ref and args.src:
        # 单对模式
        out = args.out or args.src.with_name(args.src.stem + "_aligned.obj")
        result = align_one_pair(args.ref, args.src, out, **common_kwargs)
        print(f"\n完成! fitness={result['fitness']:.4f}, RMSE={result['rmse']:.6f}")
        print(f"变换矩阵:\n{np.array(result['transform'])}")
    else:
        parser.error("请指定 --ref/--src (单对模式) 或 --ref-dir/--src-dir (批量模式)")


if __name__ == "__main__":
    main()
