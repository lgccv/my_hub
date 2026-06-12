import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass


MAX_DISTANCE = 3.2
MIN_THRESH = 0.02
STEP = 2


@dataclass
class LaserScan:
    ranges: np.ndarray
    intensities: np.ndarray
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float

def load_xyz(path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3 columns x y z, got shape {points.shape}")
    return points

def save_xyz(data, path):
    if isinstance(data, o3d.geometry.PointCloud):
        points = np.asarray(data.points)
    else:
        points = np.asarray(data, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected point cloud with shape (N, 3), got {points.shape}")
    np.savetxt(path, points, fmt="%.8f")

def xyz_to_scan_process(points: np.ndarray):
    xy = points[:,:2]
    ranges = np.linalg.norm(xy,axis=1)

    valid_mask = ranges > 1e-8
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) <2:
        raise RuntimeError("有效点太少,无法反算 angle_min 和 andle_increment")
    
    valid_angles = np.arctan2(points[valid_mask,1], points[valid_mask, 0])
    valid_angles = np.unwrap(valid_angles)

    # 用 angle = angle_min + index * angle_increment 做线性组合
    k, b = np.polyfit(valid_indices.astype(np.float64),valid_angles,1)

    angle_increment = float(k)
    angle_min = float(b)
    angle_max = float(angle_min + (len(ranges)-1) * angle_increment)

    scan = LaserScan(
        ranges = ranges.astype(np.float64),
        intensities=np.full(len(ranges),100.0,dtype = np.float64),
        angle_min = angle_min,
        angle_max = angle_max,
        angle_increment = angle_increment,
        range_min=0.0,
        range_max=float(np.max(ranges)),
    )
    return scan

def scan_to_xyz(scan: LaserScan) -> np.ndarray:
    indices = np.arange(len(scan.ranges), dtype=np.float64)
    angles = scan.angle_min + indices * scan.angle_increment

    x = scan.ranges * np.cos(angles)
    y = scan.ranges * np.sin(angles)
    z = np.zeros_like(scan.ranges)

    return np.column_stack([x, y, z])

# def plot_scan(scan: LaserScan, out_png: str):
#     indices = np.arange(len(scan.ranges), dtype=np.float64)
#     angles = scan.angle_min + indices * scan.angle_increment
#     angles_deg = np.degrees(angles)

#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         angles_deg,
#         scan.ranges,
#         s=4,
#         c=scan.ranges,
#         cmap="viridis",
#     )
#     plt.xlabel("angle (deg)")
#     plt.ylabel("range")
#     plt.title("LaserScan: angle vs range")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     plt.close()

def plot_scan_xy(scan: LaserScan, out_png: str):
    points = scan_to_xyz(scan)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=4,
        c=scan.ranges,
        cmap="viridis",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("LaserScan XY Scatter")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def filter_scan_trail_point(scan: LaserScan, max_distance):
    ranges = scan.ranges
    indexes = []

    step = 2
    min_thresh = 0.02

    if len(ranges) <= step * 2:
        return ranges, np.array([], dtype=np.int64)

    cos_increment = math.cos(scan.angle_increment * step)
    theta_thresh = math.sin(scan.angle_increment * step) / math.sin(0.17)
    scan_size = len(ranges) - step

    for i in range(step, scan_size):
        if ranges[i] == 100 or ranges[i] == 0 or ranges[i] > max_distance:
            continue

        # 这段保留 C++ 结构，但它在 C++ 里实际不会影响外层判断
        dist_direction = ranges[i + step] - ranges[i - step]
        for k in range(-step, step):
            tmp_direction = ranges[i + k + 1] - ranges[i + k]
            if dist_direction * tmp_direction <= 0:
                continue

        dist_1 = math.sqrt(
            ranges[i] * ranges[i]
            + ranges[i - step] * ranges[i - step]
            - 2 * ranges[i] * ranges[i - step] * cos_increment
        )

        dist_2 = math.sqrt(
            ranges[i] * ranges[i]
            + ranges[i + step] * ranges[i + step]
            - 2 * ranges[i] * ranges[i + step] * cos_increment
        )

        range_thresh_1 = ranges[i] * theta_thresh + min_thresh
        range_thresh_2 = ranges[i + step] * theta_thresh + min_thresh

        if dist_1 > range_thresh_1 and dist_2 > range_thresh_2:
            for j in range(-step, step + 1):
                indexes.append(i + j)

    removed_indices = np.array(sorted(set(indexes)), dtype=np.int64)

    for index in removed_indices:
        ranges[index] = 100.0

    return ranges, removed_indices

def _in_thresh(range_1, range_2, cos_increment, theta_thresh, min_thresh):
    dist = math.sqrt(
        range_1 * range_1
        + range_2 * range_2
        - 2 * range_1 * range_2 * cos_increment
    )

    range_thresh = range_1 * theta_thresh + min_thresh
    return dist <= range_thresh

def get_cluster_from_scan(scan, min_cluster_size=3):
    """
    Python 复现 C++:

    cluster_filter_.getClusterFromScan(scan_process,
                                       para_->min_continous_oba_back_scan_size);

    输入接口保持两个参数：
    1. scan
    2. min_cluster_size

    作用：
    - 按 scan 中相邻点的连续性分簇
    - 连续点数量 < min_cluster_size 的簇会被过滤
    - 被过滤的点 range 置为 0
    """
    min_thresh = 0.02
    step = 2

    ranges = scan.ranges
    scan_size = len(ranges) - step

    if scan_size <= 0:
        return scan

    cos_increment = math.cos(scan.angle_increment)
    theta_thresh = math.sin(scan.angle_increment) / math.sin(0.17)

    cos_increment_2 = math.cos(scan.angle_increment * step)
    theta_thresh_2 = math.sin(scan.angle_increment * step) / math.sin(0.17)

    class ClusterPoint:
        def __init__(self, range_value, index):
            self.range = range_value
            self.index = index
            self.continous_count = 0

    points = [ClusterPoint(ranges[i], i) for i in range(len(ranges))]

    last_point = points[0]
    index = step - 1
    minus_index = 0

    while index <= scan_size:
        if _in_thresh(
            ranges[index - 1],
            ranges[index],
            cos_increment,
            theta_thresh,
            min_thresh,
        ):
            pass

        elif _in_thresh(
            ranges[index - 1],
            ranges[index + 1],
            cos_increment_2,
            theta_thresh_2,
            min_thresh,
        ):
            points[index].range = 0
            index += 1
            minus_index += 1

        else:
            for j in range(last_point.index, index):
                points[j].continous_count = index - last_point.index - minus_index

            minus_index = 0
            last_point = points[index]

        index += 1

    if _in_thresh(
        ranges[scan_size],
        ranges[-1],
        cos_increment,
        theta_thresh,
        min_thresh,
    ):
        index += 1

    if index >= len(ranges):
        index -= 1

    for j in range(last_point.index, index + 1):
        points[j].continous_count = index - last_point.index

    # C++ 这里是 memset(ranges.data(), 0, ...)
    ranges[:] = 0.0

    for point in points:
        if point.continous_count >= min_cluster_size:
            ranges[point.index] = point.range

    return scan

def filter_scan_isolated_point(scan):
    """
    Python 复现 C++:

    ns_point_process_alg::filterScanIsolatedPoint(scan_process);

    接口一致：
    - 只传 scan 一个参数
    - 原地修改 scan.ranges
    """
    min_thresh = 0.03
    ranges = scan.ranges

    if len(ranges) <= 2:
        return scan, np.array([], dtype=np.int64)

    theta_thresh = math.sin(float(scan.angle_increment)) / math.sin(0.170)
    scan_size = len(ranges) - 1

    indexes = []

    for i in range(1, scan_size):
        if ranges[i] == 100 or ranges[i] == 0:
            continue

        dist_1 = abs(ranges[i] - ranges[i - 1])
        dist_2 = abs(ranges[i + 1] - ranges[i])

        range_thresh_1 = ranges[i - 1] * theta_thresh + min_thresh
        range_thresh_2 = ranges[i + 1] * theta_thresh + min_thresh

        if dist_1 > range_thresh_1 and dist_2 > range_thresh_2:
            indexes.append(i)

    removed_indices = np.array(indexes, dtype=np.int64)

    for index in removed_indices:
        ranges[index] = 100.0

    return scan, removed_indices


if __name__ == "__main__":
    # 1、将笛卡尔坐标反算成雷达坐标
    input_path = r"/Users/jodocls/Desktop/code/open3d/data/2026-5-28-11-42-43.147.oba_lidar_back_origin.txt"
    point_cloud_np = load_xyz(input_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

    scan = xyz_to_scan_process(point_cloud_np)

    # 保存极坐标图片
    plot_scan_xy(scan,"./result.png")
    points = scan_to_xyz(scan)

    # 2、雷达坐标过滤拖尾
    filtered_ranges, removed_indices = filter_scan_trail_point(scan, MAX_DISTANCE)

    #3、雷达坐标聚类过滤
    scan = get_cluster_from_scan(scan, 5)

    #4、过滤离群点
    scan, isolated_indices = filter_scan_isolated_point(scan)

    xyz = scan_to_xyz(scan)

    save_xyz(xyz,'./xyz.txt')