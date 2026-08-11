
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import os

from Open3D import *

@dataclass
class InstallPara:
    laser_coord_x: float
    laser_coord_y: float
    laser_coord_yaw: float
    laser_angle_min: float = math.radians(-125.0)
    laser_angle_max: float = math.radians(125)

@dataclass
class RackRegionPara:
    rack_length: float = 1.4
    rack_width: float = 0.6
    rack_leg_diameter: float = 0.3
    rack_wheel_rotate_radius: float = 0.06
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
class Cluster:
    min_index: int
    min_index_range: float
    max_index: int
    max_index_range: float
    indexes: list[int]
    points: list[np.array]

def read_points(path):
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"bad point file: {path}")
    if data.shape[1] == 2:
        data = np.column_stack([data, np.zeros(len(data))])
    return data[:, :3]


def recover_scan(points):
    ranges = np.linalg.norm(points[:,:2],axis=1)
    valid = ranges > 1e-6
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise ValueError("not enough non-zero scan points to recover scan angles")

    angles = np.unwrap(np.arctan2(points[idx,1],points[idx,0]))
    slope, intercept = np.polyfit(idx.astype(float),angles,1)
    # 每个扫描点到雷达的距离
    # 推出来的起始角度
    # 推出来的角度增量
    return ranges,float(intercept),float(slope)

def filter_out_border_points(ranges,angles,install,para):
    filtered = ranges.copy()
    invalid = (
        (filtered < para.range_min)
        | (filtered > para.range_max)
        | (angles < install.laser_angle_min)
        | (angles > install.laser_angle_max)
    )
    filtered[invalid] = 0.0
    return filtered

def rack_envelope_extra(para):
    if abs(para.rack_wheel_rotate_radius) < 1e-6:
        return 0.03
    return para.laser_std_err + para.rack_wheel_rotate_radius

def rack_rect_extents(para):
    extra = rack_envelope_extra(para)
    max_x = para.rack_length / 2.0 + para.rack_leg_diameter + extra
    max_y = para.rack_width / 2.0 + para.rack_leg_diameter + extra
    return -max_x,max_x,-max_y,max_y


def save_ray_circle_intersection_explain(angles, install, para, center_pos, radius_thresh, r0, phi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    out_path = (
        Path(__file__).resolve().parents[1]
        / "tmp/雷达通过区域过滤货架腿/ray_circle_intersection_explain.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    angle = float(angles[np.argmin(np.abs(angles - phi))])
    direction = np.array([math.cos(angle), math.sin(angle)])

    b = -2.0 * r0 * math.cos(angle - phi)
    k = r0 * r0 - radius_thresh * radius_thresh
    delta_sq = b * b - 4.0 * k
    if delta_sq < 0:
        return

    r1 = (-b + math.sqrt(delta_sq)) / 2.0
    r2 = (-b - math.sqrt(delta_sq)) / 2.0
    p_near = r2 * direction
    p_far = r1 * direction

    min_x, max_x, min_y, max_y = rack_rect_extents(para)

    def rack_to_lidar(points_rack):
        """货架/AGV坐标系下的点，变换到雷达坐标系下画图。"""
        rack_c = math.cos(para.rack_rotate)
        rack_s = math.sin(para.rack_rotate)
        agv_points = np.column_stack(
            [
                points_rack[:, 0] * rack_c - points_rack[:, 1] * rack_s,
                points_rack[:, 0] * rack_s + points_rack[:, 1] * rack_c,
            ]
        )

        laser_c = math.cos(-install.laser_coord_yaw)
        laser_s = math.sin(-install.laser_coord_yaw)
        translated_points = agv_points - np.array([install.laser_coord_x, install.laser_coord_y])
        return np.column_stack(
            [
                translated_points[:, 0] * laser_c - translated_points[:, 1] * laser_s,
                translated_points[:, 0] * laser_s + translated_points[:, 1] * laser_c,
            ]
        )

    rack_corners = np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ],
        dtype=float,
    )
    rack_corners_lidar = rack_to_lidar(rack_corners)

    leg_x = para.rack_length / 2.0
    leg_y = para.rack_width / 2.0
    leg_centers_lidar = rack_to_lidar(
        np.array(
            [
                [-leg_x, -leg_y],
                [leg_x, -leg_y],
                [leg_x, leg_y],
                [-leg_x, leg_y],
            ],
            dtype=float,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 7.2), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#555555", linewidth=1.2)
    ax.axvline(0, color="#555555", linewidth=1.2)

    check_circle = plt.Circle(center_pos, radius_thresh, fill=False, linewidth=3, color="#2ca02c")
    ax.add_patch(check_circle)
    ax.plot(
        rack_corners_lidar[:, 0],
        rack_corners_lidar[:, 1],
        color="#111111",
        linewidth=3,
        label="货架外接矩形",
    )
    for leg_center in leg_centers_lidar:
        ax.add_patch(
            plt.Circle(
                leg_center,
                para.rack_leg_diameter / 2.0,
                fill=True,
                linewidth=2,
                edgecolor="#8c564b",
                facecolor="#c49c94",
                alpha=0.8,
            )
        )
    ax.plot([0.0, p_far[0] * 1.25], [0.0, p_far[1] * 1.25], color="#1f77b4", linewidth=3)
    ax.arrow(
        0.0,
        0.0,
        center_pos[0],
        center_pos[1],
        head_width=0.10,
        head_length=0.16,
        length_includes_head=True,
        color="#ff7f0e",
        linewidth=2.5,
    )
    ax.plot(
        [center_pos[0], center_pos[0] + radius_thresh],
        [center_pos[1], center_pos[1]],
        "--",
        color="#2ca02c",
        linewidth=2,
    )

    ax.scatter([0.0], [0.0], s=80, color="red", zorder=5)
    ax.scatter([center_pos[0]], [center_pos[1]], s=80, color="#ff7f0e", zorder=5)
    ax.scatter([p_near[0], p_far[0]], [p_near[1], p_far[1]], s=95, color="#9467bd", zorder=6)
    ax.scatter(
        leg_centers_lidar[:, 0],
        leg_centers_lidar[:, 1],
        s=55,
        color="#5c4033",
        zorder=7,
    )

    ax.text(0.08, 0.08, "O 雷达原点", fontproperties=font, fontsize=13)
    ax.text(center_pos[0] - 0.55, center_pos[1] + 0.16, "C 货架中心", fontproperties=font, fontsize=13)
    ax.text(
        rack_corners_lidar[1, 0] + 0.08,
        rack_corners_lidar[1, 1] + 0.08,
        "货架外框",
        fontproperties=font,
        fontsize=13,
        color="#111111",
    )
    ax.text(
        leg_centers_lidar[0, 0] + 0.08,
        leg_centers_lidar[0, 1] - 0.22,
        "货架腿",
        fontproperties=font,
        fontsize=12,
        color="#5c4033",
    )
    ax.text(center_pos[0] * 0.48, center_pos[1] * 0.48 + 0.25, "r0", fontproperties=font, fontsize=14, color="#ff7f0e")
    ax.text(center_pos[0] + radius_thresh * 0.18, center_pos[1] - 0.18, "radius_thresh", fontsize=12, color="#2ca02c")
    ax.text(p_near[0] - 0.7, p_near[1] + 0.22, "r2 进入检查圆", fontproperties=font, fontsize=12, color="#673ab7")
    ax.text(p_far[0] + 0.12, p_far[1] + 0.12, "r1 离开检查圆", fontproperties=font, fontsize=12, color="#673ab7")

    ax.text(
        0.03,
        0.97,
        "用途：每根雷达线都和检查圆求交",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff7e6", ec="#d9a441", alpha=0.95),
    )
    ax.text(
        0.03,
        0.90,
        "result[i] = max(r1, r2)：这根线在货架检查圆里的最远距离",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eef7ff", ec="#6aa6d9", alpha=0.95),
    )

    all_points = np.vstack(
        [
            np.array([[0.0, 0.0], center_pos, p_near, p_far]),
            rack_corners_lidar,
            leg_centers_lidar,
        ]
    )
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    padding = max(0.8, radius_thresh * 0.25)
    ax.set_xlim(min_xy[0] - padding, max_xy[0] + padding)
    ax.set_ylim(min_xy[1] - padding, max_xy[1] + padding)
    ax.set_xlabel("x 雷达坐标系 / m", fontproperties=font, fontsize=13)
    ax.set_ylabel("y 雷达坐标系 / m", fontproperties=font, fontsize=13)
    ax.set_title("雷达坐标系下的货架检查圆和射线交点", fontproperties=font, fontsize=15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_rack_centered_relative_position_explain(angles, install, para, center_pos, radius_thresh, r0, phi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    out_path = (
        Path(__file__).resolve().parents[1]
        / "tmp/雷达通过区域过滤货架腿/rack_centered_relative_position_explain.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    angle = float(angles[np.argmin(np.abs(angles - phi))])
    direction = np.array([math.cos(angle), math.sin(angle)])

    b = -2.0 * r0 * math.cos(angle - phi)
    k = r0 * r0 - radius_thresh * radius_thresh
    delta_sq = b * b - 4.0 * k
    if delta_sq < 0:
        return

    r1 = (-b + math.sqrt(delta_sq)) / 2.0
    r2 = (-b - math.sqrt(delta_sq)) / 2.0
    p_near_lidar = r2 * direction
    p_far_lidar = r1 * direction

    def lidar_to_rack(points_lidar):
        laser_c = math.cos(install.laser_coord_yaw)
        laser_s = math.sin(install.laser_coord_yaw)
        points_agv = np.column_stack(
            [
                points_lidar[:, 0] * laser_c - points_lidar[:, 1] * laser_s,
                points_lidar[:, 0] * laser_s + points_lidar[:, 1] * laser_c,
            ]
        ) + np.array([install.laser_coord_x, install.laser_coord_y])

        rack_c = math.cos(para.rack_rotate)
        rack_s = math.sin(para.rack_rotate)
        return np.column_stack(
            [
                points_agv[:, 0] * rack_c + points_agv[:, 1] * rack_s,
                points_agv[:, 1] * rack_c - points_agv[:, 0] * rack_s,
            ]
        )

    lidar_origin_rack = lidar_to_rack(np.array([[0.0, 0.0]], dtype=float))[0]
    p_near_rack, p_far_rack = lidar_to_rack(np.array([p_near_lidar, p_far_lidar], dtype=float))

    min_x, max_x, min_y, max_y = rack_rect_extents(para)
    rack_corners = np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ],
        dtype=float,
    )
    leg_x = para.rack_length / 2.0
    leg_y = para.rack_width / 2.0
    leg_centers = np.array(
        [
            [-leg_x, -leg_y],
            [leg_x, -leg_y],
            [leg_x, leg_y],
            [-leg_x, leg_y],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 7.2), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#555555", linewidth=1.2)
    ax.axvline(0, color="#555555", linewidth=1.2)

    ax.add_patch(plt.Circle((0.0, 0.0), radius_thresh, fill=False, linewidth=3, color="#2ca02c"))
    ax.plot(rack_corners[:, 0], rack_corners[:, 1], color="#111111", linewidth=3)
    for leg_center in leg_centers:
        ax.add_patch(
            plt.Circle(
                leg_center,
                para.rack_leg_diameter / 2.0,
                fill=True,
                linewidth=2,
                edgecolor="#8c564b",
                facecolor="#c49c94",
                alpha=0.8,
            )
        )

    ray_extend = lidar_origin_rack + (p_far_rack - lidar_origin_rack) * 1.25
    ax.plot(
        [lidar_origin_rack[0], ray_extend[0]],
        [lidar_origin_rack[1], ray_extend[1]],
        color="#1f77b4",
        linewidth=3,
    )
    ax.arrow(
        lidar_origin_rack[0],
        lidar_origin_rack[1],
        -lidar_origin_rack[0],
        -lidar_origin_rack[1],
        head_width=0.10,
        head_length=0.16,
        length_includes_head=True,
        color="#ff7f0e",
        linewidth=2.5,
    )
    ax.plot([0.0, radius_thresh], [0.0, 0.0], "--", color="#2ca02c", linewidth=2)

    ax.scatter([0.0], [0.0], s=85, color="#ff7f0e", zorder=6)
    ax.scatter([lidar_origin_rack[0]], [lidar_origin_rack[1]], s=85, color="red", zorder=6)
    ax.scatter(
        [p_near_rack[0], p_far_rack[0]],
        [p_near_rack[1], p_far_rack[1]],
        s=95,
        color="#9467bd",
        zorder=7,
    )
    ax.scatter(leg_centers[:, 0], leg_centers[:, 1], s=55, color="#5c4033", zorder=7)

    ax.text(-0.55, 0.14, "C 货架中心", fontproperties=font, fontsize=13)
    ax.text(
        lidar_origin_rack[0] + 0.12,
        lidar_origin_rack[1] - 0.28,
        "O 雷达原点",
        fontproperties=font,
        fontsize=13,
    )
    ax.text(max_x + 0.08, max_y * 0.55, "货架外框", fontproperties=font, fontsize=13, color="#111111")
    ax.text(leg_centers[0, 0] + 0.08, leg_centers[0, 1] - 0.22, "货架腿", fontproperties=font, fontsize=12, color="#5c4033")
    ax.text(lidar_origin_rack[0] * 0.50, lidar_origin_rack[1] * 0.50 + 0.22, "r0", fontproperties=font, fontsize=14, color="#ff7f0e")
    ax.text(radius_thresh * 0.38, -0.36, "radius_thresh", fontsize=12, color="#2ca02c")
    ax.text(p_near_rack[0] - 0.70, p_near_rack[1] + 0.18, "r2 进入检查圆", fontproperties=font, fontsize=12, color="#673ab7")
    ax.text(p_far_rack[0] + 0.12, p_far_rack[1] - 0.25, "r1 离开检查圆", fontproperties=font, fontsize=12, color="#673ab7")

    ax.text(
        0.03,
        0.97,
        "货架坐标系：货架中心固定在原点",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=12,
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="#fff7e6", ec="#d9a441", alpha=0.95),
    )
    ax.text(
        0.03,
        0.90,
        "红点是雷达相对货架的位置，蓝线是同一根求交射线",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eef7ff", ec="#6aa6d9", alpha=0.95),
    )

    all_points = np.vstack(
        [
            np.array([[0.0, 0.0], lidar_origin_rack, p_near_rack, p_far_rack]),
            rack_corners,
            leg_centers,
            np.array([[radius_thresh, 0.0], [-radius_thresh, 0.0], [0.0, radius_thresh], [0.0, -radius_thresh]]),
        ]
    )
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    padding = max(0.8, radius_thresh * 0.25)
    ax.set_xlim(min_xy[0] - padding, max_xy[0] + padding)
    ax.set_ylim(min_xy[1] - padding, max_xy[1] + padding)
    ax.set_xlabel("x 货架坐标系 / m", fontproperties=font, fontsize=13)
    ax.set_ylabel("y 货架坐标系 / m", fontproperties=font, fontsize=13)
    ax.set_title("货架中心坐标系下的雷达、货架和检查圆", fontproperties=font, fontsize=15)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_center_pos_transform_explain(install, center_pos):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    out_path = (
        Path(__file__).resolve().parents[1]
        / "tmp/雷达通过区域过滤货架腿/center_pos_transform_explain.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    laser_pos_agv = np.array([install.laser_coord_x, install.laser_coord_y], dtype=float)
    translated = -laser_pos_agv
    yaw = install.laser_coord_yaw
    inverse_yaw = -yaw
    c = math.cos(inverse_yaw)
    s = math.sin(inverse_yaw)

    def draw_arrow(ax, start, vec, color, width=0.014, zorder=4):
        ax.arrow(
            start[0],
            start[1],
            vec[0],
            vec[1],
            width=width,
            head_width=0.055,
            head_length=0.080,
            length_includes_head=True,
            color=color,
            alpha=0.95,
            zorder=zorder,
        )

    def label_box(ax, xy, text, color="#333333", fc="#ffffff", ec="#999999", size=10.5, ha="left", arrow_to=None):
        if arrow_to is None:
            ax.text(
                xy[0],
                xy[1],
                text,
                fontproperties=font,
                fontsize=size,
                color=color,
                ha=ha,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, alpha=0.96),
                zorder=8,
            )
            return
        ax.annotate(
            text,
            xy=arrow_to,
            xytext=xy,
            fontproperties=font,
            fontsize=size,
            color=color,
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, alpha=0.96),
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5, shrinkA=4, shrinkB=4),
            zorder=8,
        )

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8), constrained_layout=True)

    ax = axes[0]
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#666666", linewidth=1.2)
    ax.axvline(0, color="#666666", linewidth=1.2)

    ax.plot([0.0, laser_pos_agv[0]], [0.0, 0.0], "--", color="#999999", linewidth=1.6)
    ax.plot([laser_pos_agv[0], laser_pos_agv[0]], [0.0, laser_pos_agv[1]], "--", color="#999999", linewidth=1.6)
    draw_arrow(ax, np.array([0.0, 0.0]), laser_pos_agv, "#777777")
    draw_arrow(ax, laser_pos_agv, translated, "#1f77b4")

    lidar_x = np.array([math.cos(yaw), math.sin(yaw)]) * 0.34
    lidar_y = np.array([-math.sin(yaw), math.cos(yaw)]) * 0.28
    draw_arrow(ax, laser_pos_agv, lidar_x, "#d62728", width=0.007, zorder=6)
    draw_arrow(ax, laser_pos_agv, lidar_y, "#9467bd", width=0.007, zorder=6)

    ax.scatter([0.0], [0.0], s=100, color="#ff7f0e", zorder=7)
    ax.scatter([laser_pos_agv[0]], [laser_pos_agv[1]], s=100, color="red", zorder=7)

    label_box(ax, (-0.50, 0.46), "C = (0, 0)\n货架中心 / AGV原点", color="#c26a00", fc="#fff7e6", ec="#d9a441", arrow_to=(0.0, 0.0))
    label_box(
        ax,
        (laser_pos_agv[0] + 0.06, laser_pos_agv[1] - 0.24),
        f"L = (laser_coord_x, laser_coord_y)\n  = ({laser_pos_agv[0]:.4f}, {laser_pos_agv[1]:.4f})",
        color="#b00020",
        fc="#fff1f1",
        ec="#d88a8a",
        arrow_to=laser_pos_agv,
    )
    label_box(
        ax,
        (-0.08, 0.08),
        f"laser_coord_x = {laser_pos_agv[0]:.4f}",
        color="#666666",
        fc="#f7f7f7",
        ec="#aaaaaa",
        size=10,
        arrow_to=(laser_pos_agv[0] * 0.55, 0.0),
    )
    label_box(
        ax,
        (laser_pos_agv[0] + 0.22, -0.10),
        f"laser_coord_y = {laser_pos_agv[1]:.4f}",
        color="#666666",
        fc="#f7f7f7",
        ec="#aaaaaa",
        size=10,
        arrow_to=(laser_pos_agv[0], laser_pos_agv[1] * 0.55),
    )
    label_box(
        ax,
        (-0.57, -0.47),
        (
            "translated = C - L\n"
            f"= [-laser_coord_x, -laser_coord_y]\n"
            f"= ({translated[0]:.4f}, {translated[1]:.4f})"
        ),
        color="#1f77b4",
        fc="#eef7ff",
        ec="#6aa6d9",
        arrow_to=(laser_pos_agv[0] * 0.45, laser_pos_agv[1] * 0.45),
    )
    label_box(
        ax,
        (laser_pos_agv[0] + 0.18, laser_pos_agv[1] + 0.12),
        f"laser_coord_yaw\n= {yaw:.6f} rad\n= {math.degrees(yaw):.4f} deg",
        color="#7b3fb2",
        fc="#f4edff",
        ec="#b69ad8",
        size=10,
        arrow_to=(laser_pos_agv[0] + lidar_y[0] * 0.60, laser_pos_agv[1] + lidar_y[1] * 0.60),
    )
    ax.text(laser_pos_agv[0] + lidar_x[0] * 0.78, laser_pos_agv[1] + lidar_x[1] * 0.78, "雷达x轴", fontproperties=font, fontsize=10, color="#d62728")
    ax.text(laser_pos_agv[0] + lidar_y[0] * 0.88, laser_pos_agv[1] + lidar_y[1] * 0.88, "雷达y轴", fontproperties=font, fontsize=10, color="#9467bd")

    ax.set_xlim(-0.62, 0.82)
    ax.set_ylim(-0.58, 0.62)
    ax.set_xlabel("x 货架/AGV坐标系 / m", fontproperties=font, fontsize=12)
    ax.set_ylabel("y 货架/AGV坐标系 / m", fontproperties=font, fontsize=12)
    ax.set_title("1. 在货架/AGV坐标系里：先得到 translated", fontproperties=font, fontsize=14)

    ax = axes[1]
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#666666", linewidth=1.2)
    ax.axvline(0, color="#666666", linewidth=1.2)

    ax.plot([0.0, center_pos[0]], [0.0, 0.0], "--", color="#ffbb78", linewidth=1.8)
    ax.plot([center_pos[0], center_pos[0]], [0.0, center_pos[1]], "--", color="#ffbb78", linewidth=1.8)
    draw_arrow(ax, np.array([0.0, 0.0]), center_pos, "#ff7f0e")
    ax.scatter([0.0], [0.0], s=100, color="red", zorder=7)
    ax.scatter([center_pos[0]], [center_pos[1]], s=100, color="#ff7f0e", zorder=7)

    label_box(ax, (0.05, 0.15), "O = (0, 0)\n雷达原点", color="#b00020", fc="#fff1f1", ec="#d88a8a", arrow_to=(0.0, 0.0))
    label_box(
        ax,
        (center_pos[0] - 0.48, center_pos[1] - 0.20),
        f"C in lidar = center_pos\n= ({center_pos[0]:.4f}, {center_pos[1]:.4f})",
        color="#c26a00",
        fc="#fff7e6",
        ec="#d9a441",
        arrow_to=center_pos,
    )
    label_box(
        ax,
        (center_pos[0] * 0.60, 0.12),
        f"center_pos_x\n= {center_pos[0]:.4f}",
        color="#c26a00",
        fc="#fff7e6",
        ec="#d9a441",
        size=10,
        ha="center",
        arrow_to=(center_pos[0] * 0.55, 0.0),
    )
    label_box(
        ax,
        (-0.55, 0.04),
        f"center_pos_y\n= {center_pos[1]:.4f}",
        color="#c26a00",
        fc="#fff7e6",
        ec="#d9a441",
        size=10,
        arrow_to=(center_pos[0], center_pos[1] * 0.50),
    )
    label_box(
        ax,
        (-0.77, 0.43),
        (
            "旋转关系\n"
            "center_pos = R(-yaw) * translated\n"
            "[x']   [ c  -s ] [x]\n"
            "[y'] = [ s   c ] [y]"
        ),
        color="#1f4e79",
        fc="#eef7ff",
        ec="#6aa6d9",
        size=10.5,
    )
    label_box(
        ax,
        (-0.77, -0.58),
        (
            "代入数据\n"
            f"x = translated_x = {translated[0]:.4f}\n"
            f"y = translated_y = {translated[1]:.4f}\n"
            f"c = cos(-yaw) = {c:.4f}\n"
            f"s = sin(-yaw) = {s:.4f}\n"
            f"x' = c*x - s*y = {center_pos[0]:.4f}\n"
            f"y' = s*x + c*y = {center_pos[1]:.4f}"
        ),
        color="#333333",
        fc="#f7f7f7",
        ec="#999999",
        size=10,
    )

    ax.set_xlim(-0.86, 0.55)
    ax.set_ylim(-0.72, 0.68)
    ax.set_xlabel("x 雷达坐标系 / m", fontproperties=font, fontsize=12)
    ax.set_ylabel("y 雷达坐标系 / m", fontproperties=font, fontsize=12)
    ax.set_title("2. 在雷达坐标系里：旋转后得到 center_pos", fontproperties=font, fontsize=14)

    fig.suptitle("center_pos 公式图解：每个变量和数值的位置关系", fontproperties=font, fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_check_rack_circle(angles,install,para):
    _,max_x,_,max_y = rack_rect_extents(para)
    radius_thresh = math.hypot(max_x,max_y) + para.check_dist_offset

    c = math.cos(-install.laser_coord_yaw)
    s = math.sin(-install.laser_coord_yaw)
    translated = np.array([-install.laser_coord_x,-install.laser_coord_y])
    # 货架中心在雷达坐标系下的坐标
    center_pos = np.array([c*translated[0] - s*translated[1],s*translated[0]+c*translated[1]])
    a0,b0 = center_pos
    r0_sq = a0*a0 + b0*b0
    r0 = math.sqrt(r0_sq)
    phi = math.atan2(b0,a0)

    result = np.zeros_like(angles)
    save_center_pos_transform_explain(install, center_pos)
    save_ray_circle_intersection_explain(angles, install, para, center_pos, radius_thresh, r0, phi)
    save_rack_centered_relative_position_explain(angles, install, para, center_pos, radius_thresh, r0, phi)
    for i,angle in enumerate(angles):
        b = -2.0 * r0 * math.cos(angle-phi)
        k = r0_sq - radius_thresh * radius_thresh
        delta_sq = b * b - 4.0 *k
        if delta_sq < 0:
            result[i] = 0.0
        else:
            r1 = (-b + math.sqrt(delta_sq)) / 2.0
            r2 = (-b - math.sqrt(delta_sq)) / 2.0
            result[i] = max(r1,r2,0.0)
    return result

def filter_scan_trailing_points(ranges,points_state,check_circle,angle_increment):
    step =2
    min_thresh = 0.03
    cos_increment = math.cos(angle_increment * step *2.0)
    theta_thresh = math.sin(angle_increment * step *2.0) / math.sin(math.radians(9.7))
    scan_size = len(ranges) - step
    for i in range(step,scan_size):
        if ranges[i] == 0.0 or ranges[i] > check_circle[i]:
            continue
        dist_1 = math.sqrt(max(0.0,ranges[i+step]**2+ranges[i-step]**2-2.0*ranges[i+step]*ranges[i-step]*cos_increment))
        range_thresh_1 = ranges[i] * theta_thresh + min_thresh
        if dist_1 > range_thresh_1:
            points_state[i - step : i+ step+1] = False


def extract_clusters(ranges,angles,points_state,check_circle):
    clusters = []
    curr_indexes = []
    curr_points = []

    def flush():
        nonlocal curr_indexes,curr_points
        if not curr_indexes:
            return
        if len(curr_indexes) == 1:
            ranges[curr_indexes[0]] = 0.0
        else:
            clusters.append(Cluster(
                min_index=curr_indexes[0],
                min_index_range=float(ranges[curr_indexes[0]]),
                max_index=curr_indexes[-1],
                max_index_range=float(ranges[curr_indexes[-1]]),
                indexes=curr_indexes,
                points=curr_points,
                ))
        curr_indexes = []
        curr_points = []

    for i,r in enumerate(ranges):
        if r < 1e-6 or r > check_circle[i] or not points_state[i]:
            flush()
            continue
        curr_indexes.append(i)
        curr_points.append(np.array([r*math.cos(angles[i]),r*math.sin(angles[i])]))
    flush()
    return clusters

def point_in_rotated_rect(point_agv,theta,rect):
    min_x,max_x,min_y,max_y = rect
    c = math.cos(theta)
    s = math.sin(theta)
    tmp_x = point_agv[0] * c + point_agv[1] *s
    tmp_y = point_agv[1] * c - point_agv[0] *s 
    return min_x <= tmp_x <=max_x and min_y <= tmp_y <= max_y

def transform_lidar_to_agv(points_xy,install):
    c = math.cos(install.laser_coord_yaw)
    s = math.sin(install.laser_coord_yaw)
    rot = np.array([[c,-s],[s,c]],dtype=float)
    return points_xy @ rot.T + np.array([install.laser_coord_x,install.laser_coord_y])


def find_rack_clusters(clusters,points_state,install,para):
    rect = rack_rect_extents(para)
    num_times = round(para.rack_backlash_rotate_angle / para.rectangle_angle_step)
    rack_clusters = []
    directions = [para.rack_rotate + i * para.rectangle_angle_step for i in range(-num_times,num_times+1)]

    for cluster in clusters:
        inside_count = 0
        for point_lidar in cluster.points:
            point_agv = transform_lidar_to_agv(point_lidar.reshape(1,2),install)[0]
            if any(point_in_rotated_rect(point_agv,theta,rect) for theta in directions):
                inside_count +=1
        percent = inside_count / len(cluster.points) if cluster.points else 0.0
        if percent > para.cluster_in_rectangle_thresh:
            rack_clusters.append(cluster)
    return rack_clusters


def dilate_cluster_boarder(ranges,angle_increment,cluster_index,dilate_size,direction,para):
    beg_index = cluster_index
    ref_range = ranges[beg_index]
    out_index = cluster_index
    for i in range(1,dilate_size):
        first_edge_index = beg_index + i* direction
        if first_edge_index < 0 or first_edge_index >= len(ranges):
            break
        r = ranges[first_edge_index]
        delta_dist_sq = r * r + ref_range * ref_range - 2.0 * r * ref_range * math.cos(i * angle_increment)
        if delta_dist_sq <= para.dilate_cluster_dist_thresh * para.dilate_cluster_dist_thresh:
            out_index = first_edge_index
    return out_index


def dilate_clusters(ranges,rack_clusters,angle_increment,para):
    max_index = len(ranges) -1
    delta = round(para.dilate_cluster_angle_offset / angle_increment)
    for cluster in rack_clusters:
        left_size = delta if cluster.min_index -delta > 0 else cluster.min_index
        cluster.min_index = dilate_cluster_boarder(ranges,angle_increment,cluster.min_index,left_size,-1,para)
        right_size = max_index - cluster.max_index if cluster.max_index + delta > max_index else delta
        cluster.max_index = dilate_cluster_boarder(ranges,angle_increment,cluster.max_index,right_size,1,para)


def filter_trailing_point_by_range(
    ranges: np.ndarray, min_index: int, delta_index: int, step: int, ref_range: float
) -> None:
    max_leg_range_thresh = 1.0
    for j in range(1, delta_index + 1):
        index = min_index + j * step
        if 0 <= index < len(ranges) and ranges[index] < ref_range + max_leg_range_thresh:
            ranges[index] = 100.0

def filter_rack_clusters(ranges,rack_clusters,angle_increment,para):
    max_index = len(ranges)-1
    delta = round(para.rack_leg_trailing_angle / angle_increment)
    for cluster in rack_clusters:
        left_size = delta if cluster.min_index - delta > 0 else cluster.min_index
        filter_trailing_point_by_range(ranges,cluster.min_index,left_size,-1,cluster.min_index_range)
        right_size = max_index - cluster.max_index if cluster.max_index + delta > max_index else delta
        filter_trailing_point_by_range(ranges,cluster.max_index,right_size,1,cluster.max_index_range)
        ranges[cluster.min_index : cluster.max_index +1] = 0.0


def points_from_ranges(ranges: np.ndarray, angle_min: float, angle_increment: float) -> np.ndarray:
    indexes = np.arange(len(ranges), dtype=float)
    angles = angle_min + indexes * angle_increment
    return np.column_stack([ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(ranges))])



def replay_region_filter(origin_points,install,para):
    ranges,angle_min,angle_increment = recover_scan(origin_points)
    indexes = np.arange(len(ranges),dtype=float)
    angles = angle_min + indexes * angle_increment

    work_ranges = filter_out_border_points(ranges,angles,install,para)
    check_circle = compute_check_rack_circle(angles,install,para)
    points_state = np.ones(len(work_ranges),dtype=bool)

    filter_scan_trailing_points(work_ranges,points_state,check_circle,angle_increment)
    clusters = extract_clusters(work_ranges,angles,points_state,check_circle)
    rack_clusters = find_rack_clusters(clusters,points_state,install,para)
    dilate_clusters(work_ranges,rack_clusters,angle_increment,para)
    filter_rack_clusters(work_ranges,rack_clusters,angle_increment,para)
    work_ranges[~points_state] = 0.0

    stats = {
        "angle_min": angle_min,
        "angle_increment": angle_increment,
        "origin_nonzero": int(np.count_nonzero(ranges > 1e-6)),
        "filtered_nonzero": int(np.count_nonzero(work_ranges > 1e-6)),
        "removed": int(np.count_nonzero((ranges > 1e-6) & (work_ranges <= 1e-6))),
        "clusters": len(clusters),
        "rack_clusters": len(rack_clusters),
    }
    return points_from_ranges(work_ranges, angle_min, angle_increment), stats


def filter_located_scan_trail_point(
    ranges: np.ndarray, angle_increment: float, max_distance: float = 3.2, min_thresh: float = 0.02
) -> None:
    indexes: list[int] = []
    step = 2
    cos_increment = math.cos(angle_increment * step)
    theta_thresh = math.sin(angle_increment * step) / math.sin(0.17)
    scan_size = len(ranges) - step

    for i in range(step, scan_size):
        if ranges[i] == 100.0 or ranges[i] == 0.0 or ranges[i] > max_distance:
            continue
        dist_direction = ranges[i + step] - ranges[i - step]
        direction_changed = False
        for k in range(-step, step):
            tmp_direction = ranges[i + k + 1] - ranges[i + k]
            if dist_direction * tmp_direction <= 0:
                direction_changed = True
                break
        if direction_changed:
            continue

        dist_1 = math.sqrt(
            max(
                0.0,
                ranges[i] * ranges[i]
                + ranges[i - step] * ranges[i - step]
                - 2.0 * ranges[i] * ranges[i - step] * cos_increment,
            )
        )
        dist_2 = math.sqrt(
            max(
                0.0,
                ranges[i] * ranges[i]
                + ranges[i + step] * ranges[i + step]
                - 2.0 * ranges[i] * ranges[i + step] * cos_increment,
            )
        )
        range_thresh_1 = ranges[i] * theta_thresh + min_thresh
        range_thresh_2 = ranges[i + step] * theta_thresh + min_thresh
        if dist_1 > range_thresh_1 and dist_2 > range_thresh_2:
            indexes.extend(range(i - step, i + step + 1))

    for index in indexes:
        if 0 <= index < len(ranges):
            ranges[index] = 100.0


def cluster_in_thresh(range_1: float, range_2: float, cos_increment: float, theta_thresh: float,
                      min_thresh: float) -> bool:
    dist_sq = range_1 * range_1 + range_2 * range_2 - 2.0 * range_1 * range_2 * cos_increment
    dist_1 = math.sqrt(max(0.0, dist_sq))
    range_thresh_1 = range_1 * theta_thresh + min_thresh
    return dist_1 <= range_thresh_1

def filter_cluster_by_continuous_size(
    ranges: np.ndarray, angle_increment: float, min_cluster_size: int = 3, min_thresh: float = 0.02
) -> None:
    step = 2
    cos_increment = math.cos(angle_increment)
    theta_thresh = math.sin(angle_increment) / math.sin(0.17)
    cos_increment_2 = math.cos(angle_increment * step)
    theta_thresh_2 = math.sin(angle_increment * step) / math.sin(0.17)

    scan_size = len(ranges) - step
    if scan_size <= 0:
        return

    point_ranges = ranges.copy()
    continuous_count = np.zeros(len(ranges), dtype=int)
    last_index = 0
    minus_index = 0
    index = step - 1
    while index <= scan_size:
        if cluster_in_thresh(ranges[index - 1], ranges[index], cos_increment, theta_thresh,
                             min_thresh):
            pass
        elif cluster_in_thresh(ranges[index - 1], ranges[index + 1], cos_increment_2,
                               theta_thresh_2, min_thresh):
            point_ranges[index] = 0.0
            index += 1
            minus_index += 1
        else:
            for j in range(last_index, index):
                continuous_count[j] = index - last_index - minus_index
            minus_index = 0
            last_index = index
        index += 1

    if cluster_in_thresh(ranges[scan_size], ranges[-1], cos_increment, theta_thresh, min_thresh):
        index += 1
    if index >= len(ranges):
        index -= 1
    for j in range(last_index, index + 1):
        continuous_count[j] = index - last_index

    ranges[:] = 0.0
    keep = continuous_count >= min_cluster_size
    ranges[keep] = point_ranges[keep]

def apply_located_lidar_post_filters(points):
    ranges,angle_min,angle_increment = recover_scan(points)
    filter_located_scan_trail_point(ranges,angle_increment)
    filter_cluster_by_continuous_size(ranges,angle_increment)
    return points_from_ranges(ranges,angle_min,angle_increment)

def write_points(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.6f %.6f %.6f")


def save_all_points_plane(points, ranges, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = ranges > 1e-6
    valid_indexes = np.flatnonzero(valid)

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    scatter = ax.scatter(
        points[valid, 0],
        points[valid, 1],
        c=valid_indexes,
        s=8,
        cmap="turbo",
        label="valid scan points",
    )
    ax.scatter([0.0], [0.0], c="red", s=70, marker="x", label="lidar origin / zero points")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"All lidar points in XY plane: valid={int(valid.sum())}, zero={int((~valid).sum())}"
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.78)
    colorbar.set_label("scan index")
    ax.legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scan_recover_visualization(
    points: np.ndarray,
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    path: Path,
) -> None:
    """Save a figure that explains Cartesian xy rows recovered as LaserScan polar data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    valid = ranges > 1e-6
    indexes = np.arange(len(ranges), dtype=float)
    valid_indexes = indexes[valid]
    measured_angles = np.unwrap(np.arctan2(points[valid, 1], points[valid, 0]))
    fitted_angles = angle_min + valid_indexes * angle_increment
    rebuilt_points = points_from_ranges(ranges, angle_min, angle_increment)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax_xy = axes[0, 0]
    ax_xy.scatter(points[valid, 0], points[valid, 1], s=6, c=ranges[valid], cmap="viridis")
    ax_xy.scatter([0.0], [0.0], s=50, marker="x", c="red", label="lidar origin")
    ax_xy.set_title("Cartesian points: x/y")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best")

    ax_range = axes[0, 1]
    ax_range.plot(indexes, ranges, linewidth=0.9, color="#2468a2")
    ax_range.set_title("Polar range: r = sqrt(x^2 + y^2)")
    ax_range.set_xlabel("scan index")
    ax_range.set_ylabel("range (m)")
    ax_range.grid(True, alpha=0.25)

    ax_angle = axes[1, 0]
    ax_angle.scatter(valid_indexes, measured_angles, s=6, color="#444444", label="atan2(y, x)")
    ax_angle.plot(valid_indexes, fitted_angles, color="#d1495b", linewidth=1.2,
                  label="angle_min + index * angle_increment")
    ax_angle.set_title("Polar angle recovered by linear fit")
    ax_angle.set_xlabel("scan index")
    ax_angle.set_ylabel("angle (rad)")
    ax_angle.grid(True, alpha=0.25)
    ax_angle.legend(loc="best")

    ax_compare = axes[1, 1]
    ax_compare.scatter(points[valid, 0], points[valid, 1], s=8, color="#9aa0a6", label="original xy")
    ax_compare.scatter(
        rebuilt_points[valid, 0],
        rebuilt_points[valid, 1],
        s=3,
        color="#00a676",
        label="rebuilt from r/theta",
    )
    ax_compare.set_title("Original xy vs rebuilt polar points")
    ax_compare.set_xlabel("x (m)")
    ax_compare.set_ylabel("y (m)")
    ax_compare.axis("equal")
    ax_compare.grid(True, alpha=0.25)
    ax_compare.legend(loc="best")

    fig.suptitle(
        f"recover_scan: angle_min={angle_min:.6f} rad, "
        f"angle_increment={angle_increment:.8f} rad/index",
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(origin_point_path,output_path):
    install = InstallPara(
        laser_coord_x=0.24212,
        laser_coord_y=-0.18212,
        laser_coord_yaw=math.radians(-44.7007),
        laser_angle_min= math.radians(-125.0),
        laser_angle_max=math.radians(125.0),
    )

    para = RackRegionPara(
        rack_length = 1.4,
        rack_width = 0.6,
        rack_leg_diameter=0.3,
        rack_wheel_rotate_radius=0.06,
        rack_backlash_rotate_angle=math.radians(2.0),
        rack_rotate=0.0,
        laser_std_err=0.015,
        check_dist_offset=0.5,
        rectangle_angle_step=math.radians(2.0),
        dilate_cluster_angle_offset=math.radians(1.5),
        rack_leg_trailing_angle=math.radians(2.0),
        cluster_in_rectangle_thresh=0.8,
        dilate_cluster_dist_thresh=0.03,
        range_min=0.1,
        range_max=30.0
    )

    origin = read_points(origin_point_path)
    # pcd = points_to_pcd(origin)
    # show_ply(pcd=pcd)
    origin_ranges,origin_angle_min,origin_angle_increment = recover_scan(origin)
    save_all_points_plane(origin, origin_ranges, Path(output_path) / "origin_all_points_plane.png")
    filtered,stats = replay_region_filter(origin,install,para)
    origin_filter = apply_located_lidar_post_filters(filtered)
    removed = origin.copy()
    removed[np.linalg.norm(filtered[:,:2],axis=1) > 1e-6] = 0.0
    post_removed = origin.copy()
    post_removed[np.linalg.norm(origin_filter[:,:2],axis=1) > 1e-6] = 0.0

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    write_points(os.path.join(output_path,"filtered.txt"),filtered)
    write_points(os.path.join(output_path,"removed.txt"),removed)
    write_points(os.path.join(output_path,"origin_filter.txt"),origin_filter)
    write_points(os.path.join(output_path,"origin_filter_removed.txt"),post_removed)

    save_scan_recover_visualization(
        origin,
        origin_ranges,
        origin_angle_min,
        origin_angle_increment,
        Path(output_path) / "scan_recover_visualization.png",
    )



if __name__ == "__main__":
    origin_point_path = "/Users/jodocls/Desktop/code/my_hub/4-Open3D/data/区域过滤模式/2026-7-2-13-48-59.182.locate_lidar_main_origin.txt"
    output_path = "/Users/jodocls/Desktop/code/my_hub/tmp/雷达通过区域过滤货架腿"
    main(origin_point_path,output_path)
