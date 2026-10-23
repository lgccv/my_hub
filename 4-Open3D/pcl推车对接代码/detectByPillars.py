import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Open3D import *
import math
import numpy as np

# 先聚类，然后过滤出“宽度小于 15cm、高宽比大于 2.5”的立柱。
def extract_left_right_pillars(pcd,eps=0.03,min_points =100,middle_y_abs=0.1,max_width=0.15,min_aspect_ratio=2.5):
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

        if abs(center[1]) < middle_y_abs:
            continue

        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        size = max_bound - min_bound

        x_len = float(size[0])
        y_len = float(size[1])
        z_len = float(size[2])

        aspect_ratio = z_len/ y_len

        if y_len >= max_width:
            continue

        if aspect_ratio <= min_aspect_ratio:
            continue

        pillar_infos.append({
            "label":int(label),
            "pcd": cluster_pcd,
            "points": cluster_points,
            "center": center,
            "point_num": len(cluster_points),
            "min_bound": min_bound,
            "max_bound": max_bound,
            "size" : size,
            "height":z_len,
            "aspect_ration":aspect_ratio
        })

    left_candidates = []
    right_candidates = []

    for pillar in pillar_infos:
        center_y = pillar["center"][1]
        if center_y > middle_y_abs:
            left_candidates.append(pillar)
        elif center_y < -middle_y_abs:
            right_candidates.append(pillar)

    if len(left_candidates) == 0:
        print("没有找到左立柱")
        return o3d.geometry.PointCloud(),o3d.geometry.PointCloud()

    if len(right_candidates)==0:
        print("没有找到右立柱")
        return o3d.geometry.PointCloud(),o3d.geometry.PointCloud()

    left_candidates.sort(key=lambda item: abs(item["center"][1]))
    right_candidates.sort(key=lambda item: abs(item["center"][1]))
    return left_candidates[0]["pcd"],right_candidates[0]["pcd"]


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

    # 2、通过聚类找到柱子
    # 先聚类，然后过滤出“宽度小于 15cm、高宽比大于 2.5”的立柱。
    left_pcd, right_pcd = extract_left_right_pillars(crop_pcd,eps=0.03,min_points =100,middle_y_abs=0.1,max_width=0.15,min_aspect_ratio=2.5)

    # 3、通过拟合平面去除杂点,只要内点,如果拟合到了多个平面，只要内点最多的那个平面，最后返回两个立柱
    left_plane_pcd = fit_largest_plance_inliers(left_pcd,distance_threshold=0.01,ransac_n=50,num_iterations=1000,min_plane_points=30)
    right_plane_pcd = fit_largest_plance_inliers(right_pcd,distance_threshold=0.01,ransac_n=50,num_iterations=1000,min_plane_points=30)


    # 4、在左右两个柱子取3段算平均值，分别找到2个柱子的中心点
    left_center = compute_pillar_center_by_three_heights(left_plane_pcd)
    right_center = compute_pillar_center_by_three_heights(right_plane_pcd)

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
    move_distance = docking_distance - detect_distance
    docking_center = center + pillar_normal * move_distance
    print("docking_center:",docking_center)
    print("pillar_normal:",pillar_normal)

