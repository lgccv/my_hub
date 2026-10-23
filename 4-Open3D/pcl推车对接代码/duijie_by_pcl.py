import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Open3D import *
import math
import numpy as np

def extract_left_right_pillars(pcd, eps=0.03, min_points=10, middle_y_abs=0.1):
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        )
    )

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

        # 中间立柱不要
        if abs(center[1]) < middle_y_abs:
            continue

        pillar_infos.append({
            "label": int(label),
            "pcd": cluster_pcd,
            "points": cluster_points,
            "center": center,
            "point_num": len(cluster_points)
        })

    left_candidates = []
    right_candidates = []

    for pillar in pillar_infos:
        center_y = pillar["center"][1]

        if center_y < -middle_y_abs:
            left_candidates.append(pillar)
        elif center_y > middle_y_abs:
            right_candidates.append(pillar)

    if len(left_candidates) == 0:
        raise RuntimeError("没有找到左立柱")

    if len(right_candidates) == 0:
        raise RuntimeError("没有找到右立柱")

    left_candidates.sort(key=lambda item: abs(item["center"][1]))
    right_candidates.sort(key=lambda item: abs(item["center"][1]))

    left_pillar = left_candidates[0]
    right_pillar = right_candidates[0]

    return left_pillar, right_pillar, pillar_infos

def fit_vertical_line(points):
    center_x = float(np.mean(points[:,0]))
    center_y = float(np.mean(points[:,1]))
    z_min = float(np.min(points[:,2]))
    z_max = float(np.max(points[:,2]))

    start = np.array([center_x,center_y,z_min],dtype=np.float64)
    end = np.array([center_x,center_y,z_max],dtype=np.float64)

    center = np.array([center_x, center_y, (z_min + z_max) / 2.0], dtype=np.float64)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return {
        "points": points,
        "point_num": len(points),
        "center": center,
        "direction": direction,
        "start": start,
        "end": end,
    }

def normalize(v):
    norm = np.linalg.norm(v)
    if norm ==0:
        return v
    return v / norm

def fit_line_svd(points):
    center = points.mean(axis=0)
    centered = points - center
    _,_,vh = np.linalg.svd(centered,full_matrices=False)
    direction = normalize(vh[0])
    if direction[2] <0:
        direction = -direction

    projections = centered@direction
    start = center +projections.min()* direction
    end = center+ projections.max() * direction

    return center, direction, start, end


def ransac_line(points, max_iterations=500, distance_threshold=0.01):
    best_inliers = np.empty(0, dtype=np.int64)

    for _ in range(max_iterations):
        sample = np.random.choice(len(points),size=2,replace=False)
        p0 = points[sample[0]]
        p1 = points[sample[1]]

        direction = p1 - p0
        direction_norm = np.linalg.norm(direction)

        if direction_norm ==0 :
            continue

        direction = direction / direction_norm
        distances = np.linalg.norm(np.cross(points-p0,direction),axis=1)

        inliers = np.where(distances < distance_threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 2:
        return best_inliers, None, None,None,None

    line_poitns = points[best_inliers]
    center, direction,start, end  = fit_line_svd(line_poitns)

    return best_inliers, center,direction,start,end


def extract_lines_by_ransac(points,max_lines=10,max_iterations=500,distance_threshold=0.01,min_line_points=20,remain_ratio=0.1):
    remain_points = points.copy()
    lines = []

    while len(remain_points) > len(points) * remain_ratio and len(lines) < max_lines:
        inliers, center, direction, start, end = ransac_line(remain_points,max_iterations= max_iterations,distance_threshold= distance_threshold)
        if len(inliers) < min_line_points:
            break

        line_points = remain_points[inliers]
        lines.append({
            "points": line_points,
            "point_num": len(line_points),
            "center" : center,
            "direction": direction,
            "start": start,
            "end": end
        })

        keep_mask =np.ones(len(remain_points),dtype=bool)
        keep_mask[inliers] = False
        remain_points = remain_points[keep_mask]

    return lines





def extract_pillar_side_vertical_lines(pillar_points,max_lines=20,distance_threshold=0.015,min_line_points=50,vertial_anle_threshold_deg = 15):
    lines = extract_lines_by_ransac(pillar_points,max_lines=max_lines,max_iterations=1000,distance_threshold=distance_threshold,min_line_points = min_line_points,remain_ratio = 0.05)
    vertical_lines = []

    z_axis = np.array([0.0,0.0,1.0],dtype=np.float64)
    for line in lines:
        direction = normalize(line["direction"])

        cos_theta = abs(float(np.dot(direction,z_axis)))
        cos_theta = max(-1.0,min(1.0,cos_theta))

        theta = math.degrees(math.acos(cos_theta))
        if theta > vertial_anle_threshold_deg:
            continue

        line["vertical_angle_deg"] = theta
        vertical_lines.append(line)

    if len(vertical_lines)<2:
        raise RuntimeError(f"没有找到两条竖直直线，只找到{len(vertical_lines)} 条")

    # 从竖直线中找左右两边：
    # 按 y 排序，y 最小的是左侧竖线，y 最大的是右侧竖线
    vertical_lines.sort(key=lambda item: item["center"][1])

    left_side_line = vertical_lines[0]
    right_side_line = vertical_lines[-1]

    return left_side_line, right_side_line, vertical_lines


def extract_points_between_to_vertical_lines(pillar_points,left_line,right_line,margin=0.0):
    left_center = left_line["center"]
    right_center= right_line["center"]

    left_xy = left_center[:2]
    right_xy = right_center[:2]

    side_vec = right_xy - left_xy
    side_width = np.linalg.norm(side_vec)

    if side_width == 0:
        raise RuntimeError("两条直线中心重合， 无法提取中间点云")

    side_dir = side_vec / side_width
    points_xy = pillar_points[:,:2]

    # 把每个点投影到"""
    projections = (points_xy - left_xy) @ side_dir

    between_mask = (projections >= -margin) &(projections <= side_width + margin)
    between_points = pillar_points[between_mask]
    if len(between_points) == 0:
        raise RuntimeError("两条直线之间没有点云")

    pillar_center = between_points.mean(axis=0)

    return {
        "points": between_points,
        "pcd": points_to_pcd(between_points),
        "center": pillar_center,
        "point_num": len(between_points),
        "left_line": left_line,
        "right_line": right_line,
    }

def compute_two_pillars_pose(left_center, right_center):
    left_center = np.asarray(left_center, dtype=np.float64)
    right_center = np.asarray(right_center, dtype=np.float64)

    center = (left_center + right_center) / 2.0

    width_vec = right_center - left_center
    width_vec[2] = 0.0

    pillar_y_distance = abs(width_vec[1])
    width_norm = np.linalg.norm(width_vec)

    if width_norm == 0:
        raise RuntimeError("左右立柱中心重合，无法计算角度")

    width_dir = width_vec / width_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_norm = np.cross(z_axis, width_dir)
    forward_norm = normalize(forward_norm)

    if forward_norm[0] < 0:
        forward_norm = -forward_norm

    angle = math.atan2(forward_norm[1], forward_norm[0])
    angle_deg = angle * 180.0 / math.pi

    return {
        "center": center,
        "angle": angle,
        "angle_deg": angle_deg,
        "width_dir": width_dir,
        "forward_norm": forward_norm,
        "pillar_y_distance": pillar_y_distance,
    }


def compute_pillar_center_by_three_heights(
    pillar_points,
    height_ratio=(0.25, 0.5, 0.75),
    slice_half_height=0.015,
    cube_size=0.03,
    min_slice_points=5,
    min_cube_points=3
):
    pillar_points = np.asarray(pillar_points, dtype=np.float64)

    if len(pillar_points) == 0:
        raise RuntimeError("立柱点云为空，无法计算中心")

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
        raise RuntimeError("三个高度层都没有有效点，无法计算立柱中心")

    # 5、对有效高度层的中心取平均
    refined_centers = np.asarray(
        [item["refined_center"] for item in layer_infos],
        dtype=np.float64
    )

    final_center = refined_centers.mean(axis=0)

    return {
        "center": final_center,
        "layer_infos": layer_infos,
        "refined_centers": refined_centers,
        "valid_layer_num": len(layer_infos),
        "point_num": len(pillar_points),
        "z_min": z_min,
        "z_max": z_max
    }


def fit_plane(pcd,distance_threshold,ransac_n,num_iterations):
    remain_pcd = pcd
    best_plane_model = None
    best_inliers = None
    best_score = -1.0
    best_inlier_count = 0

    while len(remain_pcd.points) >= ransac_n:
        plane_model, inlier_indices = remain_pcd.segment_plane(
            distance_threshold= distance_threshold,
            ransac_n= ransac_n,
            num_iterations=num_iterations
        )

        if len(inlier_indices) < ransac_n:
            break

        a, b, c, d = plane_model
        normal_length = math.sqrt(a*a + b*b +c*c)

        if normal_length == 0:
            break

        # 平面与X轴越垂直,平面法向量越接近X轴, score越接近1
        score = abs(a) / normal_length
        inlier_count = len(inlier_indices)

        if score > best_score or (score == best_score and inlier_count > best_inlier_count):
            best_score = score
            best_plane_model = plane_model
            best_inliers = remain_pcd.select_by_index(inlier_indices)
            best_inlier_count = inlier_count

        remain_pcd = remain_pcd.select_by_index(inlier_indices,invert=True)

    if best_plane_model is None:
        return 0, 0, 0, 0, o3d.geometry.PointCloud()

    a,b,c,d = best_plane_model
    return a,b,c,d, best_inliers

if __name__ == "__main__":
    install_x = 450
    install_y = 0
    install_z = 122
    install_roll = 0
    install_pitch = -18
    install_yaw = 0

    detect_distance = 0.7
    detect_height = 0.4

    pcl_path = r"/Users/jodocls/Desktop/code/my_hub/4-Open3D/pcl推车对接代码/data/2026-8-17-15-22-51.228.depth_camera_origin.txt"

    points = load_xyz(pcl_path)
    pcd = points_to_pcd(points)
    # show_ply(pcd)

    # 1、对点云进行坐标变换
    roll = math.radians(install_roll)
    pitch = math.radians(install_pitch)
    R = euler_to_matrix(roll,pitch,0)
    t = np.array([0.0, install_y/1000.0,install_z / 1000.0])
    pcd.rotate(R,center=(0,0,0))
    pcd.translate(t)

    # 2、对点云做裁剪
    crop_pcd = reduce_roi(pcd,detect_distance-0.2,detect_distance+0.2,-100,100,detect_height-0.1,detect_height+0.1)

    # 优化点:  首先聚类提取立柱，然后拟合平面
    # 提取立柱策略：宽度要小于15cm,  长宽比要大于2.5;

    # 3、法向量过滤 + 主平面过滤 + 离群点过滤
    estimate_filter_pcd = estimate_and_filter_normals(crop_pcd,normal_radius=0.03,angle_threshold_deg = 60.0)
    _,_,_,_,filter_pcd = fit_plane(estimate_filter_pcd,0.0,3,1000)
    radius_pcd, kept_indices = radius_filter(filter_pcd,5,0.05)

    # 4、通过聚类提取立柱，并找到左右两根立柱，中间(y<0.2)的立柱不要
    left_pillar_infos, right_pillar_infos,pillar_infos = extract_left_right_pillars(radius_pcd,eps=0.03,min_points=100,middle_y_abs=0.1)

    # 5、检查立柱的距离是否正常
    left_pillar = points_to_pcd(left_pillar_infos['points'])
    right_pillar = points_to_pcd(right_pillar_infos["points"])
    left_center_y = left_pillar_infos["center"][1]
    right_center_y = right_pillar_infos["center"][1]
    pillar_y_distance = abs(right_center_y - left_center_y)
    if pillar_y_distance < 0.3:
        print("立柱距离不对")
        exit()
    print("pillar_y_distance:",pillar_y_distance)

    # ##################  方案一
    # # 6、对左立柱提取左右两侧的两条直线
    # # 7、只要两条直线之间的点云，然后计算质心，当做立柱的中心
    # left_side_line, right_side_line, vertical_lines = extract_pillar_side_vertical_lines(left_pillar_infos["points"])
    # left_pillar_center_info = extract_points_between_to_vertical_lines(
    #     left_pillar_infos["points"],
    #     left_side_line,
    #     right_side_line,
    #     margin=0.005
    # )

    # left_side_line, right_side_line, vertical_lines = extract_pillar_side_vertical_lines(right_pillar_infos["points"])
    # right_pillar_center_info = extract_points_between_to_vertical_lines(
    #     right_pillar_infos["points"],
    #     left_side_line,
    #     right_side_line,
    #     margin=0.005
    # )

    # # 8、计算整个左右两个立柱的中心和角度
    # left_center = left_pillar_center_info["center"]
    # right_center = right_pillar_center_info["center"]

    # two_pillars_pose = compute_two_pillars_pose(left_center, right_center)
    # print("two_pillars_pose:",two_pillars_pose)
    # ##################

    # 方案二: 分三个高度提取中心点，计算中心点后再取一个3*3立方体计算中心，最后将三个高度的中心点算平均
    left_pillar_center_info = compute_pillar_center_by_three_heights(left_pillar_infos["points"],height_ratio=(0.25,0.5,0.75),slice_half_height=0.015,cube_size=0.03,min_slice_points=5,min_cube_points=3)
    right_pillar_center_info = compute_pillar_center_by_three_heights(right_pillar_infos["points"],height_ratio=(0.25,0.5,0.75),slice_half_height=0.015,cube_size=0.03,min_slice_points=5,min_cube_points=3)

    left_center = left_pillar_center_info["center"]
    right_center = right_pillar_center_info["center"]

    two_pillars_pose = compute_two_pillars_pose(left_center, right_center)

    print("left pillar center:", left_center)
    print("right pillar center:", right_center)
    print("two_pillars_pose:",two_pillars_pose)