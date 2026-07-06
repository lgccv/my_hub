#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def default_log_path() -> str:
    return "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/sensor_data_line_pallet.log"


def print_usage(program_name: str) -> None:
    print(
        f"Usage: {program_name} [sensor_data_log] [goods_width] [x_min] [x_max] [y_thresh]\n"
        f"Default log: {default_log_path()}"
    )


@dataclass
class Normal:
    normal_x: float = 0.0
    normal_y: float = 0.0
    normal_z: float = 0.0


@dataclass
class DetectResult:
    # 对齐 C++ DetectResult 中本示例会打印和写入的字段。
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    angle: float = 0.0
    width: float = 0.0
    is_available: bool = False
    normal: Normal = field(default_factory=Normal)


@dataclass
class LineDetectParam:
    x_min: float = 0.1
    x_max: float = 3.0
    voxel_size: float = 0.01
    y_thresh: float = 0.8
    goods_width: float = 1.0
    line_threshold: float = 0.3


@dataclass
class DetectionParams:
    line_params: LineDetectParam = field(default_factory=LineDetectParam)


@dataclass
class ObjectInfo:
    # 对齐 C++ ObjectInfo：一条候选直线/托盘边的信息。
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    norm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    object_width: float = 0.0


@dataclass
class CenterPoint:
    begin_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    end_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    center_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: float = 0.0


def read_sensor_data(log_path: str) -> np.ndarray | None:
    # 文件格式与 C++ readSensorData 一致：第一行是 time 和 size，后续每行是 x y z。
    path = Path(log_path)
    if not path.exists():
        return None

    points: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        if len(header) < 2:
            return None
        print(f"I: time: {float(header[0]):g}size: {float(header[1]):g}")
        if float(header[1]) == 0:
            return None

        for line in fp:
            values = line.strip().split()
            if len(values) < 3:
                continue
            points.append([float(values[0]), float(values[1]), float(values[2])])

    return np.asarray(points, dtype=np.float64)


def to_o3d_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    if len(points) > 0:
        cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def make_center_marker(center: np.ndarray) -> o3d.geometry.PointCloud:
    marker = to_o3d_cloud(
        np.array(
            [
                center,
                center + np.array([0.001, 0.0, 0.0]),
                center - np.array([0.001, 0.0, 0.0]),
                center + np.array([0.0, 0.001, 0.0]),
                center - np.array([0.0, 0.001, 0.0]),
            ],
            dtype=np.float64,
        )
    )
    marker.paint_uniform_color([0.0, 0.85, 0.1])
    return marker


def show_geometries(
    geometries: list[o3d.geometry.Geometry],
    cloud_points: np.ndarray,
    window_name: str,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    render_option.point_size = 8.0
    render_option.line_width = 2.0

    # Let Open3D compute the camera from the actual geometry bounds. Manually
    # setting front/lookat is fragile for this almost-flat z=0 laser scan.
    vis.reset_view_point(True)

    vis.run()
    vis.destroy_window()


def visualize_point_cloud(cloud_points, inliers, line_point, line_direction):
    if len(cloud_points) == 0:
        return

    inlier_points = cloud_points[inliers] if len(inliers) > 0 else np.empty((0, 3), dtype=np.float64)
    # 将向量变为单位向量
    direction = normalize(line_direction)
    direction_xy_norm = float(np.linalg.norm(direction[:2]))
    direction_xy = direction[:2] / direction_xy_norm if direction_xy_norm > 0 else np.zeros(2)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        cloud_points[:, 0],
        cloud_points[:, 1],
        s=14,
        c="#5f6368",
        label=f"cloud ({len(cloud_points)})",
    )
    if len(inlier_points) > 0:
        ax.scatter(
            inlier_points[:, 0],
            inlier_points[:, 1],
            s=28,
            c="#d93025",
            label=f"inliers ({len(inlier_points)})",
            zorder=3,
        )
    ax.scatter(
        [line_point[0]],
        [line_point[1]],
        s=80,
        c="#00b050",
        marker="x",
        linewidths=2.0,
        label="line point",
        zorder=4,
    )

    if direction_xy_norm > 0:
        if len(inlier_points) >= 2:
            projections = (inlier_points[:, :2] - line_point[:2]) @ direction_xy
            line_half_length = max(
                abs(float(projections.min())),
                abs(float(projections.max())),
                0.1,
            )
        else:
            line_half_length = max(float(np.linalg.norm(np.ptp(cloud_points[:, :2], axis=0))) * 0.15, 0.1)

        line_start = line_point[:2] - direction_xy * line_half_length
        line_end = line_point[:2] + direction_xy * line_half_length
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            color="#1a73e8",
            linewidth=2.2,
            linestyle="--",
            label="line_direction",
            zorder=2,
        )

        arrow_length = min(line_half_length * 0.6, 0.35)
        ax.quiver(
            line_point[0],
            line_point[1],
            direction_xy[0] * arrow_length,
            direction_xy[1] * arrow_length,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#1a73e8",
            width=0.006,
            headwidth=5,
            headlength=6,
            headaxislength=5,
            label=f"direction=({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})",
            zorder=5,
        )
        text_pos = line_point[:2] + direction_xy * (arrow_length * 1.1)
        ax.annotate(
            f"line_direction\n({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})",
            xy=(line_point[0], line_point[1]),
            xytext=(text_pos[0], text_pos[1]),
            color="#1a73e8",
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "#1a73e8", "lw": 1.5},
            zorder=6,
        )

    margin_x = max(float(np.ptp(cloud_points[:, 0])) * 0.08, 0.05)
    margin_y = max(float(np.ptp(cloud_points[:, 1])) * 0.08, 0.05)
    ax.set_xlim(float(cloud_points[:, 0].min()) - margin_x, float(cloud_points[:, 0].max()) + margin_x)
    ax.set_ylim(float(cloud_points[:, 1].min()) - margin_y, float(cloud_points[:, 1].max()) + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.legend(loc="best")
    ax.set_title("RANSAC point cloud")
    fig.tight_layout()
    plt.show()


def visualize_ransac_line(
    cloud_points: np.ndarray,
    inliers: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    window_name: str = "RANSAC line fitting",
) -> None:
    if len(cloud_points) == 0:
        return

    original_cloud = to_o3d_cloud(cloud_points)
    original_cloud.paint_uniform_color([0.35, 0.35, 0.35])

    inlier_points = cloud_points[inliers] if len(inliers) > 0 else np.empty((0, 3), dtype=np.float64)
    inlier_cloud = to_o3d_cloud(inlier_points)
    inlier_cloud.paint_uniform_color([0.95, 0.1, 0.05])

    direction = normalize(line_direction)
    if np.linalg.norm(direction) == 0:
        print("W: skip line visualization because line_direction is zero.")
        return

    if len(inlier_points) >= 2:
        projections = (inlier_points - line_point) @ direction
        half_length = max(abs(float(projections.min())), abs(float(projections.max())), 0.2)
    else:
        half_length = 0.1

    line_start = line_point - direction * half_length
    line_end = line_point + direction * half_length
    direction_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([line_start, line_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    direction_line.colors = o3d.utility.Vector3dVector([[0.0, 0.25, 1.0]])

    center_marker = make_center_marker(line_point)

    print(
        "I: RANSAC line visualization\n"
        f"   center: ({line_point[0]:.4f}, {line_point[1]:.4f}, {line_point[2]:.4f})\n"
        f"   direction: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})\n"
        f"   fitting inliers: {len(inliers)} / {len(cloud_points)}"
    )
    show_geometries(
        [original_cloud, inlier_cloud, direction_line, center_marker],
        cloud_points,
        window_name=window_name,
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def voxel_down_sample_pcl_style(points: np.ndarray, leaf_size: float) -> np.ndarray:
    # Open3D 的 voxel_down_sample 与 PCL VoxelGrid 的体素原点规则略有差异。
    # 这里按 PCL 的习惯用 floor(point / leaf_size) 分桶，并用桶内质心代表该体素。
    if len(points) == 0:
        return points.copy()

    voxel_indices = np.floor(points / leaf_size).astype(np.int64)
    unique_indices, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    sampled_points = np.zeros((len(unique_indices), 3), dtype=np.float64)
    counts = np.bincount(inverse)

    for axis in range(3):
        sampled_points[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts

    return sampled_points


def print_detect_result(result: DetectResult) -> None:
    print(
        f"is_available: {str(result.is_available).lower()}\n"
        f"pose: x={result.x:.4f}, y={result.y:.4f}, z={result.z:.4f}\n"
        f"angle_rad: {result.angle:.4f}\n"
        f"angle_deg: {result.angle * 180.0 / math.pi:.4f}\n"
        f"width: {result.width:.4f}\n"
        f"normal: ({result.normal.normal_x:.4f}, {result.normal.normal_y:.4f}, "
        f"{result.normal.normal_z:.4f})"
    )


class LinePalletDetector:
    def __init__(self) -> None:
        print("I: Init line detector.")
        self.debug_voxel_points_: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.debug_beside_points_: list[np.ndarray] = []
        self.debug_norm_filtered_points_: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.debug_line_points_: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.debug_line_infos_: list[ObjectInfo] = []
        self.rng = np.random.default_rng(0)

    def detect(self, cloud_points: np.ndarray, output_infos: DetectResult,
               input_param: DetectionParams) -> None:
        line_param = input_param.line_params

        # CropBox：只保留 x 范围内、且 y 在正负阈值内的点。
        mask = (
            (cloud_points[:, 0] >= line_param.x_min)
            & (cloud_points[:, 0] <= line_param.x_max)
            & (cloud_points[:, 1] >= -line_param.y_thresh)
            & (cloud_points[:, 1] <= line_param.y_thresh)
            & (cloud_points[:, 2] >= -10.0)
            & (cloud_points[:, 2] <= 10.0)
        )
        pass_through_points = cloud_points[mask]

        # VoxelGrid：下采样降低点数，同时尽量复刻 PCL 的输出。
        voxel_points = voxel_down_sample_pcl_style(pass_through_points, 0.01)
        voxel_cloud = to_o3d_cloud(voxel_points)
        self.debug_voxel_points_ = voxel_points.copy()

        if len(voxel_points) == 0:
            print("I: outlier_filter_points size:0")
            return

        # StatisticalOutlierRemoval：去掉局部邻域距离异常的孤立点。
        outlier_cloud, _ = voxel_cloud.remove_statistical_outlier(nb_neighbors=3, std_ratio=2.0)
        outlier_filter_points = np.asarray(outlier_cloud.points)
        print(f"I: outlier_filter_points size:{len(outlier_filter_points)}")

        # RANSAC 反复提取直线；随后删除法向角度偏离 x 轴太多的直线。
        line_infos = self.extract_lines_from_cloud(outlier_filter_points, max_lines=3)
        self.filter_lines(line_infos, 25)

        if len(line_infos) == 0:
            print("E: Input Error: line_infos is empty!")
            return

        print(f"I: line_infos before reshape size:{len(line_infos)}")
        # 多条线按 center.x 聚类，选 x 最小的一类，再保留点数最多的一条。
        line_infos = self.cluster_and_select_by_x(line_infos)
        line_infos = line_infos[:1]

        line = line_infos[0]
        self.debug_line_points_ = line.points.copy()

        if self.check_pallet_piers(line):
            self.debug_line_infos_ = line_infos
            if abs(line.object_width - line_param.goods_width) > line_param.line_threshold:
                print(
                    f"I: the line width is {line.object_width} out of range +-"
                    f"{line_param.line_threshold} {line_param.goods_width}"
                )

            # 输出角度沿用 C++：使用 -norm 计算 yaw。
            angle = math.atan2(-line.norm[1], -line.norm[0])

            print("I: ==================line detect result==================")
            print(f"I: norm: {line.norm[0]} {line.norm[1]} {line.norm[2]}")
            print(
                f"I: the pose in sensor: x:{line.center[0]}y:{line.center[1]}"
                f"yaw:{angle / math.pi * 180.0} degree"
            )
            print(f"I: width: {line.object_width} m")
            print("I: ==================      end       ==================")

            output_infos.angle = angle
            output_infos.is_available = True
            output_infos.width = line.object_width
            output_infos.normal.normal_x = float(line.norm[0])
            output_infos.normal.normal_y = float(line.norm[1])
            output_infos.normal.normal_z = float(line.norm[2])
            output_infos.x = float(line.center[0])
            output_infos.y = float(line.center[1])
            output_infos.z = float(line.center[2])
        else:
            print("I: ==================line detect failure==================")

    def extract_lines_from_cloud(self, input_cloud: np.ndarray, max_lines: int) -> list[ObjectInfo]:
        del max_lines  # The C++ implementation accepts this argument but does not use it.
        line_infos: list[ObjectInfo] = []
        cloud = input_cloud.copy()
        initial_size = len(input_cloud)

        # 与 C++ 一致：当剩余点数大于原始点数 10% 时继续尝试提线。
        while len(cloud) > 0.1 * initial_size:
            if len(cloud) < 2:
                break
            inliers, line_point, line_direction = self.ransac_line(
                cloud, max_iterations=1000, distance_threshold=0.015
            )
            if len(inliers) == 0:
                break

            visualize_point_cloud(cloud,inliers,line_point,line_direction)

            # PCL SACMODEL_LINE 给出的是直线方向；检测逻辑需要的是水平法向量。
            # 所以这里做 direction x Z，并强制 x 分量为正，保证方向一致。
            line_direction = np.cross(line_direction, np.array([0.0, 0.0, 1.0])) # 求法向量
            if line_direction[0] < 0:
                line_direction = -line_direction

            line_info = ObjectInfo()
            line_info.norm = normalize(line_direction)
            line_info.center = line_point
            line_info.points = cloud[inliers].copy()
            line_infos.append(line_info)

            # ExtractIndices(setNegative=true)：把已经属于当前直线的内点从剩余点云中删除。
            keep_mask = np.ones(len(cloud), dtype=bool)
            keep_mask[inliers] = False
            cloud = cloud[keep_mask]

        return line_infos

    def ransac_line(self, cloud: np.ndarray, max_iterations: int,
                    distance_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Open3D 没有直接等价于 pcl::SACMODEL_LINE 的接口，这里用 NumPy 手写线模型 RANSAC。
        best_inliers = np.empty(0, dtype=np.int64)
        best_point = np.zeros(3, dtype=np.float64)
        best_direction = np.zeros(3, dtype=np.float64)

        for _ in range(max_iterations):
            # 随机取两个点确定一条 3D 直线。
            sample = self.rng.choice(len(cloud), size=2, replace=False)
            p0 = cloud[sample[0]]
            p1 = cloud[sample[1]]
            direction = p1 - p0
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                continue
            unit_direction = direction / direction_norm
            # 点到 3D 直线的距离：|(p - p0) x direction|。
            distances = np.linalg.norm(np.cross(cloud - p0, unit_direction), axis=1)
            inliers = np.flatnonzero(distances < distance_threshold)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_point = p0
                best_direction = unit_direction

        if len(best_inliers) >= 2:
            # 对最佳内点再做一次 PCA/SVD 拟合，类似 PCL optimize coefficients 的效果。
            inlier_points = cloud[best_inliers]
            centroid = inlier_points.mean(axis=0)
            centered = inlier_points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            best_point = centroid
            best_direction = normalize(vh[0])

        return best_inliers, best_point, best_direction

    @staticmethod
    def filter_lines(lines: list[ObjectInfo], max_angle_degrees: float) -> None:
        if len(lines) == 0:
            print("E: Input Error: lines is empty!")
            return
        lines[:] = [
            # line.norm 是水平法向量；只保留法向量接近 x 轴的候选线。
            line
            for line in lines
            if abs(math.atan2(line.norm[1], line.norm[0]) * 180.0 / math.pi)
            <= max_angle_degrees
        ]

    @staticmethod
    def cluster_and_select_by_x(line_infos: list[ObjectInfo], threshold: float = 0.2) -> list[ObjectInfo]:
        clusters: list[list[ObjectInfo]] = []

        # 简单的一维聚类：用每个聚类第一条线的 center.x 作为代表值。
        for line_info in line_infos:
            found_cluster = False
            for cluster in clusters:
                if abs(cluster[0].center[0] - line_info.center[0]) < threshold:
                    cluster.append(line_info)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append([line_info])

        print(f"I:  clusters size :::::::::::::::::: {len(clusters)}")
        if len(clusters) == 0:
            return []

        # 选择 x 最小的一组候选线，通常对应最靠近相机/车体的一侧托盘边。
        min_cluster = min(clusters, key=lambda cluster: cluster[0].center[0])
        # 同一组内按内点数量降序，点最多的线更稳定。
        result = sorted(min_cluster, key=lambda line: len(line.points), reverse=True)
        print(f"I:  lineInfos size :::::::::::::::::: {len(result)}")
        return result

    def check_pallet_piers(self, line_pallet_info: ObjectInfo,
                           num_clusters_to_check: int = 3) -> bool:
        if len(line_pallet_info.points) == 0:
            return False

        # 先按 y 排序，用 y 最小/最大的两个点估算这条候选托盘边的整体宽度。
        points = line_pallet_info.points
        order = np.argsort(points[:, 1])
        points = points[order]
        line_pallet_info.points = points
        line_pallet_info.object_width = float(np.linalg.norm(points[-1] - points[0]))

        # 欧式聚类：Open3D 用 DBSCAN 实现 PCL EuclideanClusterExtraction 的近似效果。
        cloud = to_o3d_cloud(points)
        labels = np.asarray(cloud.cluster_dbscan(eps=0.06, min_points=5, print_progress=False))
        cluster_ids = [label for label in sorted(set(labels.tolist())) if label >= 0]
        piers_clouds: list[np.ndarray] = []

        for cluster_id in cluster_ids:
            cluster_points = points[labels == cluster_id]
            # 对齐 C++：每个墩聚类至少 5 个点，最多 200 个点。
            if 5 <= len(cluster_points) <= 200:
                piers_clouds.append(cluster_points.copy())

        if len(piers_clouds) < num_clusters_to_check:
            print(f"I: cluster_indices.size() {len(piers_clouds)}")
            return False

        # 先选 |y| 最小的 num_clusters_to_check 个聚类，再按 y 从小到大排列成左/中/右。
        piers_clouds.sort(key=lambda cloud_points: abs(cloud_points[0, 1]))
        piers_clouds = piers_clouds[:num_clusters_to_check]
        piers_clouds.sort(key=lambda cloud_points: cloud_points[0, 1])

        beside_points: list[np.ndarray] = []
        pier_infos: list[CenterPoint] = []
        piers_mean_y = 0.0

        for pier_cloud in piers_clouds:
            # 每个墩内部按 y 排序，用两端点计算墩宽和中心。
            pier_cloud = pier_cloud[np.argsort(pier_cloud[:, 1])]
            center = CenterPoint()
            center.begin_point = pier_cloud[0].copy()
            center.end_point = pier_cloud[-1].copy()
            center.distance = float(np.linalg.norm(center.begin_point - center.end_point))
            center.center_point = (center.begin_point + center.end_point) / 2.0

            pier_horizon_width = abs(center.begin_point[1] - center.end_point[1])
            print(
                f"I: center.center_point y: {center.center_point[1]}, "
                f"x:{center.center_point[0]}, pier_distance norm: {center.distance}, "
                f"pier_horizon_width: {pier_horizon_width}"
            )

            min_pier_width = 0.03
            # 墩太窄时认为不是稳定的托盘墩结构。
            if center.distance < min_pier_width:
                print(f"I: pier_width {center.distance} < min_pier_width")
                return False

            pier_infos.append(center)
            beside_points.extend([center.begin_point, center.end_point, center.center_point])
            piers_mean_y += center.center_point[1]

        self.debug_beside_points_ = beside_points
        # C++ 里把托盘中心设为中间墩的中心点。
        line_pallet_info.center = pier_infos[1].center_point

        # 下面两个 y 均值只用于日志观察，不参与最终判定。
        piers_mean_y /= len(piers_clouds)
        first_third_pier_mean_y = (pier_infos[0].center_point[1] + pier_infos[2].center_point[1]) * 0.5
        print(
            f"I: alg line_pallet_info.center: {pier_infos[1].center_point[1]}, "
            f"self 3_piers_mean_y: {piers_mean_y}, "
            f"self first_third_pier_mean_y: {first_third_pier_mean_y}"
        )

        left_mean_point = pier_infos[0].center_point
        middle_mean_point = pier_infos[1].center_point
        right_mean_point = pier_infos[2].center_point
        middle_to_left = left_mean_point - middle_mean_point
        middle_to_right = right_mean_point - middle_mean_point

        # 三个墩应近似共线，所以“中->左”和“中->右”的夹角应接近 180 度。
        denom = np.linalg.norm(middle_to_left) * np.linalg.norm(middle_to_right)
        if denom == 0:
            return False
        cos_value = float(np.dot(middle_to_left, middle_to_right) / denom)
        cos_value = max(-1.0, min(1.0, cos_value))
        angle_between_vector = math.acos(cos_value) * 180.0 / math.pi
        print(f"I: The angle between middle_to_left and middle_to_right is: {angle_between_vector}")

        if abs(angle_between_vector - 180.0) > 9.0:
            print(f"I: Err: The angle between middle_to_left and middle_to_right is: {angle_between_vector}")
            return False

        min_width_between_piers = 0.15
        middle_to_left_norm = float(np.linalg.norm(middle_to_left))
        middle_to_right_norm = float(np.linalg.norm(middle_to_right))
        width_rate = abs(middle_to_left_norm - middle_to_right_norm) * 2.0 / (
            middle_to_left_norm + middle_to_right_norm
        )
        # 检查左右墩间距不能太小，且左右距离相对差异不能超过 20%。
        if (
            middle_to_left_norm < min_width_between_piers
            or middle_to_right_norm < min_width_between_piers
            or width_rate > 0.2
        ):
            print(
                "I: Err: The distance between middle_to_left or middle_to_right "
                f"{middle_to_left_norm} {middle_to_left_norm}"
            )
            return False

        norm = np.zeros(3, dtype=np.float64)
        # 用 Z 轴叉乘左右墩连线方向，得到托盘在水平面内的法向量。
        if np.linalg.norm(left_mean_point - middle_mean_point) >= min_width_between_piers:
            direction = middle_mean_point - left_mean_point
            norm += normalize(np.cross(np.array([0.0, 0.0, 1.0]), direction))
        if np.linalg.norm(right_mean_point - middle_mean_point) >= min_width_between_piers:
            direction = right_mean_point - middle_mean_point
            norm += normalize(np.cross(np.array([0.0, 0.0, 1.0]), direction))

        norm = normalize(norm)
        # 统一法向量朝向，避免同一个结构因为点序不同导致方向翻转。
        line_pallet_info.norm = norm if float(np.dot(middle_mean_point, norm)) > 0 else -norm
        return True


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] in {"-h", "--help"}:
        print_usage(argv[0])
        return 0

    log_path = argv[1] if len(argv) > 1 else default_log_path()
    cloud = read_sensor_data(log_path)
    if cloud is None:
        print(f"E: Failed to read sensor data: {log_path}")
        print_usage(argv[0])
        return 1

    detect_params = DetectionParams()

    print(
        f"log_path: {log_path}\n"
        f"cloud_points: {len(cloud)}\n"
        f"goods_width: {detect_params.line_params.goods_width}\n"
        f"x_range: [{detect_params.line_params.x_min}, {detect_params.line_params.x_max}]\n"
        f"y_thresh: {detect_params.line_params.y_thresh}"
    )

    detector = LinePalletDetector()
    detect_result = DetectResult()
    detector.detect(cloud, detect_result, detect_params)

    print_detect_result(detect_result)
    print(
        f"debug_voxel_points: {len(detector.debug_voxel_points_)}\n"
        f"debug_beside_points: {len(detector.debug_beside_points_)}\n"
        f"debug_line_points: {len(detector.debug_line_points_)}\n"
        f"debug_line_infos: {len(detector.debug_line_infos_)}"
    )

    return 0 if detect_result.is_available else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
