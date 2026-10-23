import sys
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Open3D import *
import math
import numpy as np

def euclidean_cluster_indices(pcd, tolerance=0.03, min_points=100):
    """按照 PCL EuclideanClusterExtraction 的连通规则生成聚类索引。"""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []

    tree = o3d.geometry.KDTreeFlann(pcd)
    processed = np.zeros(len(points), dtype=bool)
    clusters = []

    for start_index in range(len(points)):
        if processed[start_index]:
            continue

        processed[start_index] = True
        seed_queue = deque([start_index])
        cluster = []

        while seed_queue:
            point_index = seed_queue.popleft()
            cluster.append(point_index)
            _, neighbor_indices, _ = tree.search_radius_vector_3d(
                points[point_index], tolerance
            )
            for neighbor_index in neighbor_indices:
                neighbor_index = int(neighbor_index)
                if not processed[neighbor_index]:
                    processed[neighbor_index] = True
                    seed_queue.append(neighbor_index)

        if len(cluster) >= min_points:
            clusters.append(np.asarray(cluster, dtype=np.int64))

    return clusters


def build_pillar_info(pcd):
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        return None

    point_num = len(points)
    center = points.mean(axis=0, dtype=np.float32)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    height = float(max_bound[2] - min_bound[2])

    y_values = np.sort(points[:, 1])
    last_index = point_num - 1
    lower_index = int(math.floor(last_index * 0.05))
    upper_index = int(math.ceil(last_index * 0.95))
    robust_y_width = float(y_values[upper_index] - y_values[lower_index])

    centered_points = points - center
    covariance = (centered_points.T @ centered_points) / np.float32(point_num)
    _, eigenvectors = np.linalg.eigh(covariance)
    verticality = abs(float(eigenvectors[2, -1]))

    return {
        "pcd": pcd,
        "points": points,
        "center": center,
        "point_num": point_num,
        "min_bound": min_bound,
        "max_bound": max_bound,
        "size": max_bound - min_bound,
        "robust_y_width": robust_y_width,
        "verticality": verticality,
        "height": height
    }


def extract_left_right_pillars(
    pcd,
    eps=0.03,
    min_points=100,
    max_robust_y_width=0.18,
    min_verticality=0.9,
    max_center_z_difference=0.05,
    max_height = 0.2,
):
    left_candidates = []
    right_candidates = []

    for indices in euclidean_cluster_indices(pcd, eps, min_points):
        cluster_pcd = pcd.select_by_index(indices.tolist())
        info = build_pillar_info(cluster_pcd)
        if info is None:
            continue

        center = info["center"]
        print(
            "立柱候选簇:",
            "center=", center,
            "points=", info["point_num"],
            "robust_y_width=", info["robust_y_width"],
            "verticality=", info["verticality"],
        )

        if info["point_num"] < min_points:
            continue
        if info["verticality"] < min_verticality:
            continue
        if info["robust_y_width"] > max_robust_y_width:
            continue
        if info["height"] < max_height - 0.05:
            continue

        if center[1] > 0:
            left_candidates.append(info)
        elif center[1] < 0:
            right_candidates.append(info)

    if not left_candidates:
        print("没有找到左立柱")
        return None, None
    if not right_candidates:
        print("没有找到右立柱")
        return None, None

    left_candidates.sort(key=lambda item: abs(float(item["center"][1])))
    right_candidates.sort(key=lambda item: abs(float(item["center"][1])))

    left_pillar = left_candidates[0]
    right_pillar = right_candidates[0]

    print(
        "选中的左右立柱:",
        "left=", left_pillar["center"],
        "right=", right_pillar["center"],
        "center_z_difference=",
        abs(float(left_pillar["center"][2] - right_pillar["center"][2])),
    )

    return left_pillar, right_pillar


def _fit_xy_line_ransac(
    xy_points,
    distance_threshold,
    num_iterations,
    random_generator,
):
    """在一组 XY 点中拟合一条直线，并返回精修后的内点索引和方向。"""
    point_num = len(xy_points)
    if point_num < 2:
        return np.empty(0, dtype=np.int64), None

    best_inlier_indices = np.empty(0, dtype=np.int64)

    for _ in range(num_iterations):
        sample_indices = random_generator.choice(point_num, size=2, replace=False)
        point_a = xy_points[sample_indices[0]]
        point_b = xy_points[sample_indices[1]]
        direction = point_b - point_a
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= np.finfo(np.float64).eps:
            continue

        direction /= direction_norm
        point_offsets = xy_points - point_a
        distances = np.abs(
            point_offsets[:, 0] * direction[1]
            - point_offsets[:, 1] * direction[0]
        )
        inlier_indices = np.flatnonzero(distances <= distance_threshold)
        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices

    if len(best_inlier_indices) < 2:
        return best_inlier_indices, None

    # 使用 RANSAC 内点做一次 PCA 精修，再重新计算内点。
    inlier_xy = xy_points[best_inlier_indices]
    line_center = inlier_xy.mean(axis=0)
    covariance = np.cov(inlier_xy - line_center, rowvar=False, bias=True)
    _, eigenvectors = np.linalg.eigh(covariance)
    line_direction = eigenvectors[:, -1]
    line_direction /= np.linalg.norm(line_direction)

    point_offsets = xy_points - line_center
    distances = np.abs(
        point_offsets[:, 0] * line_direction[1]
        - point_offsets[:, 1] * line_direction[0]
    )
    refined_inlier_indices = np.flatnonzero(distances <= distance_threshold)
    return refined_inlier_indices, line_direction


def fit_best_xy_line_inliers(
    pcd,
    expected_width_direction,
    distance_threshold=0.002,
    num_iterations=1000,
    min_input_points=50,
    min_line_points=100,
    max_direction_angle_deg=30.0,
    min_z_span=0.15,
    random_seed=42,
    cloud_name="pillar",
):
    """筛选与双柱宽度方向一致的 XY 主直线，并返回对应的原始 XYZ 点。"""
    if pcd is None or len(pcd.points) < min_input_points:
        point_num = 0 if pcd is None else len(pcd.points)
        print(cloud_name, "XY直线筛选失败: 输入点数不足", point_num)
        return o3d.geometry.PointCloud()

    points = np.asarray(pcd.points, dtype=np.float64)
    expected_direction = np.asarray(
        expected_width_direction, dtype=np.float64
    )[:2].copy()
    expected_direction_norm = np.linalg.norm(expected_direction)
    if expected_direction_norm <= np.finfo(np.float64).eps:
        print(cloud_name, "XY直线筛选失败: 期望宽度方向长度为0")
        return o3d.geometry.PointCloud()
    expected_direction /= expected_direction_norm

    random_generator = np.random.default_rng(random_seed)
    remaining_indices = np.arange(len(points), dtype=np.int64)
    best_original_indices = np.empty(0, dtype=np.int64)
    best_angle_deg = float("inf")
    candidate_index = 0

    while len(remaining_indices) >= min_input_points:
        remaining_xy = points[remaining_indices, :2]
        local_inlier_indices, line_direction = _fit_xy_line_ransac(
            remaining_xy,
            distance_threshold,
            num_iterations,
            random_generator,
        )
        if line_direction is None or len(local_inlier_indices) < min_line_points:
            break

        candidate_index += 1
        original_inlier_indices = remaining_indices[local_inlier_indices]
        candidate_points = points[original_inlier_indices]
        z_span = float(np.ptp(candidate_points[:, 2]))
        direction_cosine = float(
            np.clip(abs(np.dot(line_direction, expected_direction)), 0.0, 1.0)
        )
        direction_angle_deg = math.degrees(math.acos(direction_cosine))
        direction_valid = direction_angle_deg <= max_direction_angle_deg
        height_valid = z_span >= min_z_span

        print(
            cloud_name,
            "XY直线候选:",
            "index=", candidate_index,
            "points=", len(original_inlier_indices),
            "direction=", line_direction,
            "angle_deg=", direction_angle_deg,
            "z_span=", z_span,
            "valid=", direction_valid and height_valid,
        )

        is_better = len(original_inlier_indices) > len(best_original_indices)
        is_equal_but_straighter = (
            len(original_inlier_indices) == len(best_original_indices)
            and direction_angle_deg < best_angle_deg
        )
        if direction_valid and height_valid and (is_better or is_equal_but_straighter):
            best_original_indices = original_inlier_indices
            best_angle_deg = direction_angle_deg

        keep_mask = np.ones(len(remaining_indices), dtype=bool)
        keep_mask[local_inlier_indices] = False
        remaining_indices = remaining_indices[keep_mask]

    if len(best_original_indices) == 0:
        print(cloud_name, "XY直线筛选失败: 没有满足方向和高度条件的直线")
        return o3d.geometry.PointCloud()

    filtered_pcd = pcd.select_by_index(best_original_indices.tolist())
    print(
        cloud_name,
        "XY直线筛选完成:",
        "before=", len(points),
        "after=", len(filtered_pcd.points),
        "angle_deg=", best_angle_deg,
    )
    return filtered_pcd



def fit_largest_plance_inliers(
    pcd,
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000,
    min_input_points=50,
    min_plane_points=30,
):
    if pcd is None or len(pcd.points) < min_input_points:
        return o3d.geometry.PointCloud()

    remain_pcd = pcd
    best_plane_pcd = o3d.geometry.PointCloud()
    best_inlier_count = 0

    while len(remain_pcd.points) >= min_input_points:
        _, inlier_indices = remain_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if len(inlier_indices) < min_plane_points:
            break

        plane_pcd = remain_pcd.select_by_index(inlier_indices)
        inlier_count = len(inlier_indices)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane_pcd = plane_pcd

        remain_pcd = remain_pcd.select_by_index(inlier_indices, invert=True)

    return best_plane_pcd



def normalize(v):
    norm = np.linalg.norm(v)
    if norm ==0:
        return v
    return v / norm

def compute_two_pillars_pose(left_center, right_center):
    if len(left_center)==0:
        print("左立柱中心计算失败")
        return [],[],[],[]

    if len(right_center)==0:
        print("右立柱中心计算失败")
        return [],[],[],[]


    left_center = np.asarray(left_center, dtype=np.float64)
    right_center = np.asarray(right_center, dtype=np.float64)

    center = (left_center + right_center) / 2.0

    width_vec = right_center - left_center
    width_vec[2] = 0.0

    pillar_distance = np.linalg.norm(width_vec)  # 立柱的宽度

    if pillar_distance < 1e-6:
        print("左右立柱中心重合，无法计算角度")
        return [],[],[],[]

    width_dir = width_vec / pillar_distance

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_dir = np.cross(z_axis, width_dir)
    forward_norm = np.linalg.norm(forward_dir)
    if forward_norm < 1e-6:
        print("对接朝向计算失败")
        return [],[],[],[]

    forward_dir = forward_dir /forward_norm

    if forward_dir[0] < 0:
        forward_dir = -forward_dir

    angle = math.atan2(forward_dir[1], forward_dir[0])
    angle_deg = angle * 180.0 / math.pi

    return center,angle_deg,pillar_distance,forward_dir


def compute_pillar_center_by_three_heights(
    pcd,
    height_ratio=(0.25, 0.5, 0.75),
    slice_half_height=0.015,
    cube_size=0.03,
    min_slice_points=5,
    min_cube_points=3
):
    pillar_points = np.asarray(pcd.points, dtype=np.float64)

    if len(pillar_points) == 0:
        print("立柱点云为空，无法计算中心")
        return []

    z_min = float(np.min(pillar_points[:, 2]))
    z_max = float(np.max(pillar_points[:, 2]))

    cube_half = cube_size / 2.0
    layer_infos = []

    for ratio in height_ratio:
        target_z = z_min + (z_max - z_min) * ratio

        # 1、在指定高度附近取一个薄层
        slice_mask = np.abs(pillar_points[:, 2] - target_z) <= slice_half_height
        slice_points = pillar_points[slice_mask]

        # 当前高度层点数不足，跳过，不报错
        if len(slice_points) < min_slice_points:
            print("当前高度层为:",target_z,"当前高度层数点不足:",len(slice_points),"阈值为:",min_slice_points)
            continue

        # 2、先用薄层点云计算一个粗中心
        coarse_center = slice_points.mean(axis=0)

        # 3、以粗中心为中心，取 cube_size * cube_size * cube_size 立方体
        cube_mask = (
            (np.abs(pillar_points[:, 0] - coarse_center[0]) <= cube_half)
            & (np.abs(pillar_points[:, 1] - coarse_center[1]) <= cube_half)
            & (np.abs(pillar_points[:, 2] - coarse_center[2]) <= cube_half)
        )

        cube_points = pillar_points[cube_mask]

        # 当前高度层立方体内点数不足，跳过，不报错
        if len(cube_points) < min_cube_points:
            print("当前高度层为:",target_z,"当前立方体点不足:",len(cube_points),"阈值为:",min_cube_points)
            continue

        # 4、用立方体内点云重新计算精修中心
        refined_center = cube_points.mean(axis=0)

        layer_infos.append({
            "ratio": ratio,
            "target_z": target_z,
            "coarse_center": coarse_center,
            "refined_center": refined_center,
            "slice_points": slice_points,
            "cube_points": cube_points,
            "slice_points_num": len(slice_points),
            "cube_points_num": len(cube_points)
        })

    # 三个高度都没有有效结果，才报错
    if len(layer_infos) == 0:
        print("三个层都没有高度")
        return []

    # 5、对有效高度层的中心取平均
    refined_centers = np.asarray(
        [item["refined_center"] for item in layer_infos],
        dtype=np.float64
    )

    print("检测到层数是:",len(refined_centers),"数据分别为:",refined_centers)

    final_center = refined_centers.mean(axis=0)
    print("三层得到的平均点为:",final_center)

    return final_center


if __name__ == "__main__":
    detect_distance = 0.9
    detect_height = 0.5

    docking_distance = 0.6

    pcl_path = r"/Users/jodocls/Downloads/10.10.71.198/202608242000/sensor_data_two_piers_pcl.txt"

    points = load_xyz(pcl_path)
    pcd = points_to_pcd(points)

    # 1、裁剪ROI点云
    crop_pcd = reduce_roi(pcd,detect_distance-0.2,detect_distance+0.2,-100,100,detect_height-0.1,detect_height+0.1)

    # 2、通过欧式聚类、PCA 竖直度和稳健 Y 宽度筛选左右立柱。
    left_pillar, right_pillar = extract_left_right_pillars(
        crop_pcd,
        eps=0.01,
        min_points=100,
        max_robust_y_width=0.18,
        min_verticality=0.9,
        max_center_z_difference=0.05,
        max_height=0.2
    )
    if left_pillar is None or right_pillar is None:
        print("左右立柱聚类失败")
        sys.exit(1)

    width_direction = (
        np.asarray(right_pillar["center"], dtype=np.float64)
        - np.asarray(left_pillar["center"], dtype=np.float64)
    )
    width_direction[2] = 0.0
    if np.linalg.norm(width_direction) <= np.finfo(np.float64).eps:
        print("左右立柱原始中心重合，无法计算XY期望方向")
        sys.exit(1)

    # 3、在XY平面筛选与双柱宽度方向一致的主直线，去除立柱侧面。
    left_xy_line_pcd = fit_best_xy_line_inliers(
        left_pillar["pcd"],
        width_direction,
        distance_threshold=0.002,
        num_iterations=1000,
        min_input_points=50,
        min_line_points=100,
        max_direction_angle_deg=30.0,
        min_z_span=0.15,
        random_seed=42,
        cloud_name="左立柱",
    )
    right_xy_line_pcd = fit_best_xy_line_inliers(
        right_pillar["pcd"],
        width_direction,
        distance_threshold=0.002,
        num_iterations=1000,
        min_input_points=50,
        min_line_points=100,
        max_direction_angle_deg=30.0,
        min_z_span=0.15,
        random_seed=43,
        cloud_name="右立柱",
    )
    if len(left_xy_line_pcd.points) == 0 or len(right_xy_line_pcd.points) == 0:
        print("左右立柱XY主直线筛选失败")
        sys.exit(1)

    # 4、对XY筛选结果拟合三维平面，进一步去除偏离正面的杂点。
    left_plane_pcd = fit_largest_plance_inliers(
        left_xy_line_pcd,
        distance_threshold=0.002,
        ransac_n=3,
        num_iterations=1000,
        min_input_points=50,
        min_plane_points=30,
    )
    right_plane_pcd = fit_largest_plance_inliers(
        right_xy_line_pcd,
        distance_threshold=0.002,
        ransac_n=3,
        num_iterations=1000,
        min_input_points=50,
        min_plane_points=30,
    )
    if len(left_plane_pcd.points) == 0 or len(right_plane_pcd.points) == 0:
        print("左右立柱三维平面拟合失败")
        sys.exit(1)

    print(
        "三维平面精修完成:",
        "left_before=", len(left_xy_line_pcd.points),
        "left_after=", len(left_plane_pcd.points),
        "right_before=", len(right_xy_line_pcd.points),
        "right_after=", len(right_plane_pcd.points),
    )


    # 5、在左右两个柱子取3段算平均值，分别找到2个柱子的中心点
    left_center = compute_pillar_center_by_three_heights(left_plane_pcd)
    right_center = compute_pillar_center_by_three_heights(right_plane_pcd)

    # 6、通过两个中心点计算对接的中心点和角度
    center,angle,pillar_width,pillar_normal = compute_two_pillars_pose(left_center, right_center)
    if len(center) ==0:
        print("两个立柱位姿计算失败")
        sys.exit(1)

    print("left pillar center:", left_center)
    print("right pillar center:", right_center)
    print("center_pose:",center)
    print("angle:",angle)
    print("pillar_with:",pillar_width)
    print("pillar_normal:",pillar_normal)

    # 7、根据偏移量进行偏移
    move_distance = docking_distance - detect_distance
    docking_center = center + pillar_normal * move_distance
    print("docking_center:",docking_center)
    print("pillar_normal:",pillar_normal)
