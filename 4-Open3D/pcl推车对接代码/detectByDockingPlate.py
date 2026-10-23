import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Open3D import *
import math
import numpy as np


def extract_docking_pillars(pcd,eps=0.03,min_points=100,min_width=0.1):
    labels = np.asarray(pcd.cluster_dbscan(eps=eps,min_points=min_points,print_progress=False))
    pillar_infos = []
    for label in np.unique(labels):
        if label < 0:
            continue

        indices = np.where(labels == label)[0]
        cluster_pcd = pcd.select_by_index(indices)
        cluster_points = np.asarray(cluster_pcd.points)

        if len(cluster_points) == 0:
            continue

        center = cluster_points.mean(axis=0)

        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        size = max_bound - min_bound

        x_len = float(size[0])
        y_len = float(size[1])
        z_len = float(size[2])

        width = y_len

        if width <= min_width:
            continue

        pillar_infos.append({
            "label": int(label),
            "pcd": cluster_pcd,
            "points": cluster_points,
            "center": center,
            "point_num": len(cluster_points),
            "min_bound": min_bound,
            "max_bound": max_bound,
            "size": size,
            "x_len": x_len,
            "y_len": y_len,
            "z_len": z_len,
            "width": width
        })

    if len(pillar_infos) == 0:
        print("没有找到符合宽度条件的柱子")
        return o3d.geometry.PointCloud()

    pillar_infos.sort(key=lambda item: abs(item["center"][1]))

    return pillar_infos[0]["pcd"]


def fit_largest_plance_inliers(pcd,distance_threshold=0.01,ransac_n=100,num_iterations=1000,min_plane_points =30):
    remain_pcd = pcd
    best_plane_model = None
    best_plane_pcd = None
    best_inlier_count = 0

    while len(remain_pcd.points) >= ransac_n:
        plane_model,inlier_indices = remain_pcd.segment_plane(
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
            best_plane_model = plane_model
            best_plane_pcd = plane_pcd

        remain_pcd = remain_pcd.select_by_index(inlier_indices, invert=True)

    if best_plane_pcd is None:
        return None, o3d.geometry.PointCloud()

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


def compute_center_by_width_percent(pcd,detect_distance,detect_height,min_cube_points =3,percent=0.05,cube_size=0.03,x_half_range=0.1):
    points = np.asarray(pcd.points,dtype=np.float64)
    if len(points) ==0:
        print("filter_pcd 为空，无法计算左边中心")
        return []

    y_min = float(np.min(points[:,1]))
    y_max = float(np.max(points[:,1]))
    y_width = y_max - y_min


    if y_width <= 1e-6:
        print("filter_pcd 宽度异常，无法计算左边中心")
        return []

    target_y = y_max - y_width * percent

    target_point = np.array(
        [detect_distance, target_y, detect_height],
        dtype=np.float64
    )

    cube_half = cube_size / 2.0
    cube_mask = (
        (np.abs(points[:, 0] - target_point[0]) <= x_half_range)
        & (np.abs(points[:, 1] - target_point[1]) <= cube_half)
        & (np.abs(points[:, 2] - target_point[2]) <= cube_half)
    )

    cube_points = points[cube_mask]

    if len(cube_points) < min_cube_points:
        print("左边 5% 位置附近立方体点数不足")
        return []

    center = cube_points.mean(axis=0)
    return center




if __name__ == "__main__":
    detect_distance = 0.67
    detect_height = -0.2


    pcl_path = r"/Users/jodocls/Desktop/code/my_hub/4-Open3D/pcl推车对接代码/data/2026-8-17-15-22-51.228.depth_camera_origin.txt"

    points = load_xyz(pcl_path)
    pcd = points_to_pcd(points)

    # 1、裁剪ROI点云
    crop_pcd = reduce_roi(pcd,detect_distance-0.2,detect_distance+0.2,-100,100,detect_height-0.02,detect_height+0.02)

    # 2、通过聚类找到柱子,选择abs(y)最小的那一个，并且宽度要大于10cm
    docking_pcd = extract_docking_pillars(crop_pcd,eps=0.03,min_points =100,min_width=0.1)

    # 3、通过拟合平面去除躁点
    filter_pcd = fit_largest_plance_inliers(docking_pcd,distance_threshold=0.01,ransac_n=50,num_iterations=1000,min_plane_points=30)

    # 4、计算filter_pcd的整体宽度，左边5%的地方取3*3的立方体作为左边的中心
    left_center = compute_center_by_width_percent(filter_pcd,detect_distance,detect_height,min_cube_points =3,percent=0.05,cube_size=0.03,x_half_range=0.1)
    right_center = compute_center_by_width_percent(filter_pcd,detect_distance,detect_height,min_cube_points =3,percent=0.95,cube_size=0.03,x_half_range=0.1)

    # 5、通过两个中心点计算对接的中心点和角度
    center,angle,pillar_width,pillar_normal = compute_two_pillars_pose(left_center, right_center)
    if len(center) ==0:
        print("两个立柱位姿计算失败")
        exit()

    print("left pillar center:", left_center)
    print("right pillar center:", right_center)
    print("center_pose:",center)
    print("angle:",angle)
    print("pillar_with:",pillar_width)
    print("pillar_normal:",pillar_normal)

    # 6、根据偏移量进行偏移
    move_distance = detect_distance - detect_distance
    docking_center = center + pillar_normal * move_distance
    print("docking_center:",docking_center)
    print("pillar_normal:",pillar_normal)

