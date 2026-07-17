#!/usr/bin/env python3
"""Replay located-lidar rack legs filtering from recorded scan point files.

This script mirrors the legs branch in modules/laser/laser_filter/rack_filter.hpp:

  RackLegsFilter::computeRackInfo()
    -> build four rack-leg circular-sector regions in lidar coordinates
  BaseRackFilter::filterRack()
    -> filterOutBoarderPoint()
    -> filterScanTrailingPoint()
    -> extractClusters()
    -> RackLegsFilter::findRackCluster()
    -> dilateClusters()
    -> filterRackClusters()

It is intentionally fixed to the main-lidar frame recorded in:
  sros_货架腿模式/log/avoid_obs_2.0/2026-7-2-13-47-6.182.locate_lidar_main_origin.txt
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from 单线雷达通过货架腿过滤货架腿 import (
    Cluster,
    InstallPara,
    apply_located_lidar_post_filters,
    compute_check_rack_circle,
    dilate_clusters,
    extract_clusters,
    filter_out_border_points,
    filter_rack_clusters,
    filter_scan_trailing_points,
    points_from_ranges,
    read_points,
    recover_scan,
    to_pcd,
    visualize,
    write_points,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data/货架腿过滤模式"
FIXED_SENSOR = "main"
FIXED_INPUT = DATA_DIR / "2026-7-2-13-47-6.182.locate_lidar_main_origin.txt"
FIXED_COMPARE = DATA_DIR / "2026-7-2-13-47-6.182.locate_lidar_main_origin_filter.txt"
FIXED_OUT_DIR = ROOT_DIR / "tmp/rack_region_replay_main"


@dataclass
class RackLegsPara:
    rack_length: float = 1.4
    rack_width: float = 0.6
    rack_leg_diameter: float = 0.3
    # In this code path onUpdateRackInfo() rebuilds rack::RackPara without wheel radius.
    # The log confirms leg_radius=0.18 = 0.3/2 + 0.03.
    rack_wheel_rotate_radius: float = 0.0
    rack_backlash_rotate_angle: float = math.radians(2.0)
    rack_rotate: float = 0.0
    laser_std_err: float = 0.015
    check_dist_offset: float = 0.5
    rectangle_angle_step: float = math.radians(2.0)
    dilate_cluster_angle_offset: float = math.radians(1.5)
    rack_leg_trailing_angle: float = math.radians(2.0)
    cluster_in_rectangle_thresh: float = 0.8
    dilate_cluster_dist_thresh: float = 0.03
    range_min: float = 0.1
    range_max: float = 30.0


@dataclass
class Circle:
    min_theta: float
    max_theta: float
    radius_min: float
    radius_max: float

    def contains(self, point_xy: np.ndarray) -> bool:
        dist = float(np.linalg.norm(point_xy))
        if not (dist > self.radius_min and dist < self.radius_max):
            return False
        angle = math.atan2(float(point_xy[1]), float(point_xy[0]))
        return angle >= self.min_theta and angle <= self.max_theta


def fixed_main_install() -> InstallPara:
    return InstallPara(
        laser_coord_x=0.24212,
        laser_coord_y=-0.18212,
        laser_coord_yaw=math.radians(-44.7007),
        laser_angle_min=math.radians(-125.0),
        laser_angle_max=math.radians(125.0),
    )


def rack_envelope_extra(para: RackLegsPara) -> float:
    if abs(para.rack_wheel_rotate_radius) < 1e-6:
        return 0.03
    return para.laser_std_err + para.rack_wheel_rotate_radius


def rack_rect_extents(para: RackLegsPara) -> tuple[float, float, float, float]:
    extra = rack_envelope_extra(para)
    max_x = para.rack_length / 2.0 + para.rack_leg_diameter + extra
    max_y = para.rack_width / 2.0 + para.rack_leg_diameter + extra
    return -max_x, max_x, -max_y, max_y


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_agv_to_lidar(point_agv: np.ndarray, install: InstallPara) -> np.ndarray:
    translated = point_agv - np.array([install.laser_coord_x, install.laser_coord_y])
    return translated @ rotation_matrix(-install.laser_coord_yaw).T


def transform_lidar_for_direction(point_lidar: np.ndarray, install: InstallPara,
                                  direction: float) -> np.ndarray:
    # C++: curr_tf = laser_tf.inverse() * rotate_tf(-direction) * laser_tf
    point_agv = point_lidar @ rotation_matrix(install.laser_coord_yaw).T
    point_agv += np.array([install.laser_coord_x, install.laser_coord_y])
    point_agv = point_agv @ rotation_matrix(-direction).T
    return transform_agv_to_lidar(point_agv, install)


def compute_leg_circles(install: InstallPara, para: RackLegsPara) -> list[Circle]:
    center_x = para.rack_length / 2.0
    center_y = para.rack_width / 2.0
    leg_base_radius = para.rack_leg_diameter / 2.0
    leg_radius = leg_base_radius + rack_envelope_extra(para)

    rack_leg_center_points = [
        np.array([center_x, center_y], dtype=float),
        np.array([center_x, -center_y], dtype=float),
        np.array([-center_x, -center_y], dtype=float),
        np.array([-center_x, center_y], dtype=float),
    ]

    num_times = round(para.rack_backlash_rotate_angle / para.rectangle_angle_step)
    if num_times < 1:
        num_times = 1

    circles: list[Circle] = []
    for rack_center in rack_leg_center_points:
        for i in range(-num_times, num_times + 1):
            rotate_delta_angle = float(i) * para.rectangle_angle_step
            leg_agv = rack_center @ rotation_matrix(rotate_delta_angle).T
            leg = transform_agv_to_lidar(leg_agv, install)
            leg_dist = float(np.linalg.norm(leg))
            if leg_dist <= leg_radius:
                continue
            delta_angle = math.asin(leg_radius / leg_dist)
            center_angle = math.atan2(float(leg[1]), float(leg[0]))
            circles.append(
                Circle(
                    min_theta=center_angle - delta_angle,
                    max_theta=center_angle + delta_angle,
                    radius_min=leg_dist - leg_radius,
                    radius_max=leg_dist + leg_radius,
                )
            )
    return circles


def find_rack_leg_clusters(
    clusters: list[Cluster],
    points_state: np.ndarray,
    install: InstallPara,
    para: RackLegsPara,
    circles: list[Circle],
) -> list[Cluster]:
    rack_clusters: list[Cluster] = []
    for cluster in clusters:
        cluster_in_rack_count = 0
        for point_lidar, index in zip(cluster.points, cluster.indexes):
            curr_point = transform_lidar_for_direction(point_lidar, install, para.rack_rotate)
            for circle in circles:
                if not circle.contains(curr_point):
                    continue
                cluster_in_rack_count += 1
                points_state[index] = False
                break

        percent = cluster_in_rack_count / len(cluster.points) if cluster.points else 0.0
        if percent > para.cluster_in_rectangle_thresh:
            rack_clusters.append(cluster)
    return rack_clusters


def replay_legs_filter(
    origin_points: np.ndarray, install: InstallPara, para: RackLegsPara
) -> tuple[np.ndarray, dict[str, int | float]]:
    ranges, angle_min, angle_increment = recover_scan(origin_points)
    angles = angle_min + np.arange(len(ranges), dtype=float) * angle_increment

    work_ranges = filter_out_border_points(ranges, angles, install, para)
    check_circle = compute_check_rack_circle(angles, install, para)
    points_state = np.ones(len(work_ranges), dtype=bool)
    filter_scan_trailing_points(work_ranges, points_state, check_circle, angle_increment)
    clusters = extract_clusters(work_ranges, angles, points_state, check_circle)
    circles = compute_leg_circles(install, para)
    rack_clusters = find_rack_leg_clusters(clusters, points_state, install, para, circles)
    dilate_clusters(work_ranges, rack_clusters, angle_increment, para)
    filter_rack_clusters(work_ranges, rack_clusters, angle_increment, para)
    work_ranges[~points_state] = 0.0

    stats = {
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "origin_nonzero": int(np.count_nonzero(ranges > 1e-6)),
        "filtered_nonzero": int(np.count_nonzero(work_ranges > 1e-6)),
        "removed": int(np.count_nonzero((ranges > 1e-6) & (work_ranges <= 1e-6))),
        "clusters": len(clusters),
        "rack_clusters": len(rack_clusters),
        "leg_circles": len(circles),
        "leg_radius": para.rack_leg_diameter / 2.0 + rack_envelope_extra(para),
    }
    return points_from_ranges(work_ranges, angle_min, angle_increment), stats


def compare_masks(label: str, expected: np.ndarray, replay: np.ndarray) -> None:
    expected_range = np.linalg.norm(expected[:, :2], axis=1)
    replay_range = np.linalg.norm(replay[:, :2], axis=1)
    expected_zero = expected_range <= 1e-6
    replay_zero = replay_range <= 1e-6
    expected_100 = np.isclose(expected_range, 100.0, atol=1e-3)
    replay_100 = np.isclose(replay_range, 100.0, atol=1e-3)
    print(label)
    print(f"  same_zero_mask: {int(np.count_nonzero(expected_zero == replay_zero))}/{len(replay)}")
    print(f"  expected_zero: {int(np.count_nonzero(expected_zero))}")
    print(f"  replay_zero: {int(np.count_nonzero(replay_zero))}")
    print(f"  mismatch_zero_mask: {int(np.count_nonzero(expected_zero != replay_zero))}")
    print(f"  same_100_mask: {int(np.count_nonzero(expected_100 == replay_100))}/{len(replay)}")
    print(f"  expected_100: {int(np.count_nonzero(expected_100))}")
    print(f"  replay_100: {int(np.count_nonzero(replay_100))}")
    print(f"  mismatch_100_mask: {int(np.count_nonzero(expected_100 != replay_100))}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis", action="store_true", help="open Open3D viewer after fixed replay")
    args = parser.parse_args()

    install = fixed_main_install()
    para = RackLegsPara()
    origin = read_points(FIXED_INPUT)
    filtered, stats = replay_legs_filter(origin, install, para)
    origin_filter = apply_located_lidar_post_filters(filtered)
    removed = origin.copy()
    removed[np.linalg.norm(filtered[:, :2], axis=1) > 1e-6] = 0.0
    post_removed = origin.copy()
    post_removed[np.linalg.norm(origin_filter[:, :2], axis=1) > 1e-6] = 0.0

    FIXED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_points(FIXED_OUT_DIR / f"{FIXED_SENSOR}_legs_replay_filtered.txt", filtered)
    write_points(FIXED_OUT_DIR / f"{FIXED_SENSOR}_legs_replay_removed.txt", removed)
    write_points(FIXED_OUT_DIR / f"{FIXED_SENSOR}_legs_replay_origin_filter.txt", origin_filter)
    write_points(FIXED_OUT_DIR / f"{FIXED_SENSOR}_legs_replay_origin_filter_removed.txt", post_removed)

    print("Fixed main-lidar rack-legs replay:")
    print(f"  input: {FIXED_INPUT}")
    print(f"  compare_rack_filter: {FIXED_COMPARE_RACK}")
    print(f"  compare_origin_filter: {FIXED_COMPARE_ORIGIN_FILTER}")
    print(f"  out_dir: {FIXED_OUT_DIR}")
    print("Replay stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  rect_extents(min_x,max_x,min_y,max_y): {rack_rect_extents(para)}")

    compare_masks("Compare against recorded rack_filter:", read_points(FIXED_COMPARE_RACK), filtered)
    compare_masks(
        "Compare against recorded origin_filter:",
        read_points(FIXED_COMPARE_ORIGIN_FILTER),
        origin_filter,
    )

    try:
        import open3d as o3d
    except ImportError as exc:
        print(f"Open3D is not installed: {exc}")
        print("Run with base env: /home/standard/miniconda3/condabin/conda run -n base python ...")
        return 1

    pcd_specs = (
        ("origin", origin, (0.45, 0.45, 0.45)),
        ("filtered", filtered, (0.1, 0.65, 1.0)),
        ("removed", removed, (1.0, 0.15, 0.1)),
        ("origin_filter", origin_filter, (0.0, 0.8, 0.25)),
        ("origin_filter_removed", post_removed, (1.0, 0.4, 0.0)),
    )
    for name, points, color in pcd_specs:
        path = FIXED_OUT_DIR / f"{FIXED_SENSOR}_{name}.pcd"
        o3d.io.write_point_cloud(str(path), to_pcd(points, color))
        point_count = len(o3d.io.read_point_cloud(str(path)).points)
        print(f"  {path}: {point_count} points")

    if args.vis:
        visualize(origin, origin_filter, post_removed, install, para)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
