# 先转pitch,还是先转height   答案:先pitch/roll  后height
# 要用什么样的pitch和height,才可以将点云转到水平
#  
from Open3D import *
import open3d as o3d
import math
import os

def rotation_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0,0.0,0.0],
            [0.0,c,-s],
            [0.0,s,c]
        ],dtype=np.float64,
    )

def rotation_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )

def rotation_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fit_ground_plane(pcd):
    crop_pcd = reduce_roi(pcd, 0, 1.5, -0.5, 0.5, -1, 1)
    if len(crop_pcd.points) < 3:
        raise RuntimeError("ROI 内点数少于 3 个，无法拟合平面")

    plane_model, inliers = crop_pcd.segment_plane(
        distance_threshold=0.015,
        ransac_n=3,
        num_iterations=3000,
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # 统一让法向量朝向 +Z，避免 pitch 差 180 度
    if normal[2] < 0.0:
        normal = -normal
        plane_model = [-a, -b, -c, -d]

    inlier_pcd = crop_pcd.select_by_index(inliers)
    return normal, plane_model, inlier_pcd


def estimate_best_pitch(point_cloud,roll,output_path="./stage1_roll_pitch.txt"):
    normal,plane_model,_=fit_ground_plane(point_cloud)
    roll_rad = math.radians(roll)
    normal_after_roll = rotation_x(roll_rad) @ normal

    if normal_after_roll[2] < 0.0:
        normal_after_roll = -normal_after_roll

    # Ry(pitch) 后要求法向量 x 分量为 0:
    # nx*cos(pitch) + nz*sin(pitch) = 0
    pitch_rad = math.atan2(-normal_after_roll[0], normal_after_roll[2])
    pitch_deg = math.degrees(pitch_rad)

    corrected_pcd = o3d.geometry.PointCloud(point_cloud)
    corrected_pcd.rotate(
        rotation_y(pitch_rad) @ rotation_x(roll_rad),
        center=(0.0, 0.0, 0.0),
    )

    save_xyz(corrected_pcd, output_path)

    print("最好的 pitch 值为:", pitch_deg)
    print("保存 pitch 矫正点云:", os.path.abspath(output_path))

    return pitch_deg, corrected_pcd


def estimate_best_height(point_cloud, output_path="./stage2_roll_pitch_height.txt"):
    _, _, ground_inlier_pcd = fit_ground_plane(point_cloud)

    ground_points = np.asarray(ground_inlier_pcd.points)
    ground_z = float(np.median(ground_points[:, 2]))

    # 把地面平移到 z=0
    height_m = -ground_z
    height_mm = height_m * 1000.0

    corrected_pcd = o3d.geometry.PointCloud(point_cloud)
    corrected_pcd.translate((0.0, 0.0, height_m))

    save_xyz(corrected_pcd, output_path)

    print("最好的 height 值为:", height_m, "m")
    print("最好的 height 值为:", height_mm, "mm")
    print("保存 height 矫正点云:", os.path.abspath(output_path))

    return height_m, corrected_pcd

if __name__ == "__main__":
    txt_path = r"/Users/jodocls/Downloads/sros 98/log/avoid_obs_2.0/2026-10-8-15-12-31.280.depth_camera_origin.txt"
    points = load_xyz(txt_path)
    pcd = points_to_pcd(points)
    estimate_pcd = o3d.geometry.PointCloud(pcd)

    # 计算出最好的pitch值，保存好矫正平的点云
    pitch_deg, pitch_pcd = estimate_best_pitch(estimate_pcd, roll=0)
    height_m, height_pcd = estimate_best_height(pitch_pcd)

    print("最终 pitch:", pitch_deg)
    print("最终 height:", height_m, "m")

