from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import open3d as o3d

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data/区域过滤模式"
FIXED_SENSOR = "main"
FIXED_INPUT = DATA_DIR / "2026-7-2-13-48-59.182.locate_lidar_main_origin.txt"
FIXED_COMPARE = DATA_DIR / "2026-7-2-13-48-59.182.locate_lidar_main_rack_filter.txt"
FIXED_OUT_DIR = ROOT_DIR / "tmp/rack_region_replay_main"

@dataclass
class InstallPara:
    laser_coord_x: float
    laser_coord_y: float
    laser_coord_yaw: float
    laser_angle_min: float = math.radians(-125.0)
    laser_angle_max: float = math.radians(125.0)

@dataclass
class RackRegionPara:
    rack_length: float = 1.4
    rack_width: float = 0.6
    rack_leg_diameter: float =0.3
    rack_wheel_rotate_radius: float =0.06
    rack_backlash_rotate_angle: float = math.radians(2.0)
    rack_rotate: float=0.0
    laser_std_err: float = 0.015
    check_dist_offset: float =0.5
    rectangle_angle_step: float = math.radians(2.0)
    dilate_cluster_angle_offset: float = math.radians(1.5)
    rack_leg_trailing_angle: float = math.radians(2.0)
    cluster_in_rectagnle_thresh: float = 0.8
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
    points: list[np.ndarray]


def fixed_main_install():
    return InstallPara(
        laser_coord_x=0.24212,
        laser_coord_y=-0.18212,
        laser_coord_yaw=math.radians(-44.7007),
        laser_angle_min=math.radians(-125.0),
        laser_angle_max=math.radians(125.0))



def fixed_region_para():
    return RackRegionPara(
        rack_length = 1.4,
        rack_width=0.6,
        rack_leg_diameter=0.3,
        rack_wheel_rotate_radius=0.06,
        rack_backlash_rotate_angle=math.radians(2.0),
        rack_rotate=0.0,
        laser_std_err = 0.015,
        check_dist_offset=0.5,
        rectangle_angle_step = math.radians(2.0),
        dilate_cluster_angle_offset=math.radians(1.5),
        rack_leg_trailing_angle=math.radians(2.0),
        cluster_in_rectagnle_thresh=0.8,
        dilate_cluster_dist_thresh=0.03,
        range_min=0.1,
        range_max=30.0
    )

def read_points(path):
    data = np.loadtxt(path,dtype=float)
    if data.ndim == 1:
        data = data.reshape(1,-1)
    if data.shape[1] < 2:
        raise ValueError(f"bad point file:{path}")
    if data.shape[1] == 2:
        data = np.column_stack([data,np.zeros(len(data))])
    return data[:,:3]

def write_point(path,points):
    path.parent.mkdir(parents=True,exist_ok = True)
    np.savetxt(path,points,fmt="%.6f %.6f %.6f")

def recover_scan(points):
    ranges = np.linalg.norm(points[:,:2],axis=1)
    valid = ranges > 1e-6
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise ValueError("not enough non-zero scan points to recover scan angles")
    angles = np.unwrap(np.arctan2(points[idx,1],points[idx,0]))
    slop,intercept = np.polyfit(idx.astype(float),angles,1)
    return ranges,float(intercept),float(slop)

def filter_out_border_points(ranges,angles,install,para):
    filtered = ranges.copy()
    invalid = ((filtered < para.range_min) | (filtered > para.range_max) | (angles < install.laser_angle_min) | (angles > install.laser_angle_max))
    filtered[invalid] = 0.0
    return filtered

def rack_envelope_extra(para):
    if abs(para.rack_wheel_rotate_radius) < 1e-6:
        return 0.03
    return para.laser_std_err + para.rack_wheel_rotate_radius

def rack_rect_extents(para):
    extra = rack_envelope_extra(para)
    max_x = para.rack_length /2.0 + para.rack_leg_diameter + extra
    max_y = para.rack_width /2.0 + para.rack_leg_diameter + extra
    return -max_x,max_x,-max_y,max_y


def compute_check_rack_circle(angles,install,para):
    _,max_x,_,max_y = rack_rect_extents(para)
    radius_thresh = math.hypot(max_x,max_y) + para.check_dist_offset

    c = math.cos(-install.laser_coord_yaw)
    s = math.sin(-install.laser_coord_yaw)
    translated = np.array([-install.laser_coord_x,-install.laser_coord_y])
    center_pose = np.array([c * translated[0]-s*translated[1],s*translated[0]+c*translated[1]])

    a0, b0 = center_pose
    r0_sq = a0*a0 + b0*b0
    r0 = math.sqrt(r0_sq)
    phi = math.atan2(b0,a0)

    result = np.zeros_like(angles)
    for i,angle in enumerate(angles):
        b = -2.0 * r0*math.cos(angle-phi)
        k = r0_sq - radius_thresh * radius_thresh
        delta_sq = b *b -4.0*k
        if delta_sq <0:
            result[i] = 0.0
        else:
            r1 = (-b + math.sqrt(delta_sq)) /2.0
            r2 = (-b - math.sqrt(delta_sq)) /2.0
            result[i] = max(r1,r2,0.0)
    return result

def filter_scan_trailing_points(ranges,point_state,check_circle,angle_increment):
    step = 2
    min_thresh = 0.03
    cos_increment = math.cos(angle_increment * step *2.0)
    theta_thresh = math.sin(angle_increment*step*2.0)/math.sin(math.radians(9.7))
    scan_size = len(ranges)- step
    for i in range(step,scan_size):
        if ranges[i] ==0.0 or ranges[i] > check_circle[i]:
            continue
        dist_1 = math.sqrt(max(0.0,ranges[i+step]**2+ranges[i-step]**2-2.0*ranges[i+step]*ranges[i-step]*cos_increment))
        range_thresh_1 = ranges[i]*theta_thresh+min_thresh
        if dist_1 > range_thresh_1:
            point_state[i-step :i+step +1] = False

def extract_clusters(ranges,angles,points_state,check_circle):
    clusters = []
    curr_indexes = []
    curr_points = []

    def flush():
        nonlocal curr_indexes,curr_points
        if not curr_indexes:
            return
        if len(curr_indexes) ==1:
            ranges[curr_indexes[0]] = 0.0
        else:
            clusters.append(Cluster(
                min_index=curr_indexes[0],
                min_index_range=float(ranges[curr_indexes[0]]),
                max_index = curr_indexes[-1],
                max_index_range=float(ranges[curr_indexes[-1]]),
                indexes=curr_indexes,
                points = curr_points,
            ))

    for i,r in enumerate(ranges):
        if r<1e-6 or r> check_circle[i] or not points_state[i]:
            flush()
            continue
        curr_indexes.append(i)
        curr_points.append(np.array([r*math.cos(angles[i]),r*math.sin(angles[i])]))
    flush()
    return clusters


def transform_lidar_to_agv(points_xy,install):
    c = math.cos(install.laser_coord_yaw)
    s = math.sin(install.laser_coord_yaw)
    rot = np.array([[c,-s],[s,c]],dtype=float)
    return points_xy @ rot.T + np.array([install.laser_coord_x,install.laser_coord_y])

def point_in_rotated_rect(point_agv,theta,rect):
    min_x,max_x,min_y,max_y = rect
    c = math.cos(theta)
    s = math.sin(theta)
    tmp_x = point_agv[0] *c + point_agv[1] * s
    tmp_y = point_agv[1] *c - point_agv[0] * s
    return min_x <=tmp_x <= max_x and min_y <= tmp_y <=max_y



def find_rack_clusters(clusters,points_state,install,para):
    rect = rack_rect_extents(para)
    num_times = round(para.rack_backlash_rotate_angle / para.rectanle_angle_step)
    rack_clusters = []
    directions = [para.rack_rotate +i*para.rectangle_angle_step for i in range(-num_times,num_times+1)]

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
        first_edge_index = beg_index +i*direction
        if first_edge_index < 0 or first_edge_index >= len(ranges):
            break
        r = ranges[first_edge_index]
        delta_dist_sq = r * r +ref_range * ref_range -2.0*r*ref_range *math.cos(i*angle_increment)
        if delta_dist_sq <= para.dilate_cluster_dist_thresh * para.dilate_cluster_dist_thresh:
            out_index = first_edge_index
    return out_index




def dilate_clusters(ranges,rack_clusters,angle_increment,para):
    max_index = len(ranges) -1
    delta = round(para.dilate_cluster_angle_offset /angle_increment)
    for cluster in rack_clusters:
        left_size = delta if cluster.min_index - delta > 0 else cluster.min_index
        cluster.min_index = dilate_cluster_boarder(ranges,angle_increment,cluster.min_index,left_size,-1,para)
        right_size = max_index - cluster.max_index if cluster.max_index + delta > max_index else delta
        cluster.max_index = dilate_cluster_boarder(ranges,angle_increment,cluster.max_index,right_size,1,para)


def filter_trailing_point_by_range(ranges,min_index,delta_index,step,ref_range):
    max_leg_range_thresh = 1.0
    for j in range(1,delta_index +1):
        index = min_index +j*step
        if 0 <= index < len(ranges) and ranges[index] < ref_range + max_leg_range_thresh:
            ranges[index] = 100.0

def filter_rack_clusters(ranges,rack_clusters,angle_increment,para):
    max_index = len(ranges) -1
    delta = round(para.rack_leg_trailing_angle /angle_increment)
    for cluster in rack_clusters:
        left_size = delta if cluster.min_index - delta >0 else cluster.min_index
        filter_trailing_point_by_range(ranges,cluster.min_index,left_size,-1,cluster.min_index_range)
        right_size = max_index - cluster.max_index if cluster.max_index + delta > max_index else delta
        filter_trailing_point_by_range(ranges,cluster.max_index,right_size,1,cluster.max_index_range)
        ranges[cluster.min_index:cluster.max_index+1] = 0.0

def points_from_ranges(ranges,angle_min,angle_increment):
    indexes = np.arange(len(ranges),dtype=float)
    angles = angle_min +indexes * angle_increment
    return np.column_stack([ranges*np.cos(angles),ranges*np.sin(angles),np.zeros(len(ranges))])


def replay_region_filter(origin_points,install,para):
    ranges, angle_min,angle_increment = recover_scan(origin_points)
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
        "angle_min" : angle_min,
        "angle_increment" : angle_increment,
        "origin_nonzero": int(np.count_nonzero(ranges > 1e-6)),
        "filtered_nonzero": int(np.count_nonzero(work_ranges > 1e-6)),
        "removed": int(np.count_nonzero((ranges > 1e-6) & (work_ranges <= 1e-6))),
        "clusters": len(clusters),
        "rack_clusters": len(rack_clusters),
    }

    return points_from_ranges(work_ranges,angle_min,angle_increment), stats

def filter_located_scan_trail_point(ranges,angle_increment,max_distance,min_thresh):
    indexes = []
    step =2
    cos_increment = math.cos(angle_increment * step)
    theta_thresh = math.sin(angle_increment * step) /math.sin(0.17)
    scan_size = len(ranges) - step

    for i in range(step,scan_size):
        if ranges[i] ==100.0 or ranges[i] ==0.0 or ranges[i] > max_distance:
            continue
        dist_direction = ranges[i+ step]- ranges[i-step]
        direction_changed = False
        for k in range(-step,step):
            tmp_direction = ranges[i+k+1]-ranges[i+k]
            if dist_direction *tmp_direction <=0:
                direction_changed=True
                break
        if direction_changed:
            continue

        dist_1 = math.sqrt(max(0.0,ranges[i]*ranges[i] + ranges[i-step]*step[i-step]-2.0*range[i]*range[i-step]*cos_increment))
        dist_2 = math.sqrt(max(0.0,ranges[i]*ranges[i]+range[i+step]*ranges[i+step]-2.0*ranges[i]*ranges[i+step]*cos_increment))

        range_thresh_1 = ranges[i]*theta_thresh +min_thresh
        range_thresh_2 = ranges[i+ step]*theta_thresh + min_thresh
        if dist_1 > range_thresh_1 and dist_2 > range_thresh_2:
            indexes.extend(range(i-step,i+step+1))

    for index in indexes:
        if 0 <= index < len(ranges):
            ranges[index] = 100.0


def cluster_in_thresh(range_1,range_2,cos_increment,theta_thresh,min_thresh):
    dist_sq = range_1 * range_1 +range_2 * range_2 -2.0*range_1*range_2*cos_increment
    dist_1 = math.sqrt(max(0.0,dist_sq))
    range_thresh_1 = range_1 *theta_thresh +min_thresh
    return dist_1 <= range_thresh_1


def filter_cluster_by_continous_size(ranges,angle_increment,min_cluster_size,min_thresh):
    step =2
    cos_increment = math.cos(angle_increment)
    theta_thresh = math.cos(angle_increment)/math.sin(0.17)
    cos_increment_2 = math.cos(angle_increment *step)
    theta_thresh_2 = math.sin(angle_increment*step)/math.sin(0.17)

    scan_size = len(ranges) - step
    if scan_size <=0:
        return 
    
    point_ranges = ranges.copy()
    continuous_count = np.zeros(len(ranges),dtype=int)
    last_index = 0
    minus_index = 0
    index = step -1
    while index <= scan_size:
        if cluster_in_thresh(ranges[index-1],ranges[index],cos_increment,theta_thresh,min_thresh):
            pass
        elif cluster_in_thresh(ranges[index-1],ranges[index+1],cos_increment_2,theta_thresh_2,min_thresh):
            point_ranges[index] = 0.0
            index +=1
            minus_index +=1
        else:
            for j in range(last_index,index):
                continuous_count[j] = index - last_index - minus_index
            minus_index = 0
            last_index = index
        inde +=1

        if cluster_in_thresh(ranges[scan_size],ranges[-1],cos_increment,theta_thresh,min_thresh):
            index +=1
        if index >= len(ranges):
            index -=1
        for j in range(last_index,index +1):
            continuous_count[j] = index - last_index

        ranges[:] = 0.0
        keep = continuous_count >= min_cluster_size
        ranges[keep] = point_ranges[keep]


def apply_located_lidar_post_filters(points):
    ranges,angle_min,angle_increment = recover_scan(points)
    filter_located_scan_trail_point(ranges,angle_increment)
    filter_cluster_by_continous_size(ranges,angle_increment)
    return points_from_ranges(ranges,angle_min,angle_increment)


def to_pcd(points: np.ndarray, color: tuple[float, float, float]):
    import open3d as o3d

    nonzero = np.linalg.norm(points[:, :2], axis=1) > 1e-6
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[nonzero])
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (int(nonzero.sum()), 1)))
    return pcd

def make_line_set(points: list[tuple[float, float, float]], lines: list[tuple[int, int]], color: tuple[float, float, float]):
    import open3d as o3d

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(lines), 1)))
    return line_set

def rack_rect_lines(install: InstallPara, para: RackRegionPara) -> list:
    min_x, max_x, min_y, max_y = rack_rect_extents(para)
    corners = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype=float
    )
    c = math.cos(para.rack_rotate)
    s = math.sin(para.rack_rotate)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    corners = corners @ rot.T
    pts = [(float(x), float(y), 0.02) for x, y in corners]
    return make_line_set(pts, [(0, 1), (1, 2), (2, 3), (3, 0)], (1.0, 0.8, 0.0))


def visualize(origin,filtered,removed,install,para):
    import open3d as o3d
    geoms = [
        to_pcd(origin,(0.45,0.45,0.45)),
        to_pcd(filtered,(0.1,0.65,1.0)),
        to_pcd(removed,(1.0,0.15,0.1)),
        rack_rect_lines(install,para),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]

    o3d.visualization.draw_geometries(geoms,window_name = "rack_region filter replay")
def main():
    install = fixed_main_install()
    para = fixed_region_para()

    origin = read_points(FIXED_INPUT)
    filtered,stats = replay_region_filter(origin,install,para)
    origin_filter = apply_located_lidar_post_filters(filtered)

    removed = origin.copy()
    removed[np.linalg.norm(filtered[:,:2],axis=1) > 1e-6] =0.0
    post_removed = origin.copy()
    post_removed[np.linalg.norm(origin_filter[:,:2],axis=1) > 1e-6] = 0.0

    FIXED_OUT_DIR.mkdir(parents=True,exist_ok=True)
    write_point(FIXED_OUT_DIR/f"{FIXED_SENSOR}_region_replay_filtered.txt",filtered)
    write_point(FIXED_OUT_DIR/f"{FIXED_SENSOR}_region_replay_removed.txt",removed)
    write_point(FIXED_OUT_DIR/f"{FIXED_SENSOR}_region_replay_origin_filter.txt",origin_filter)
    write_point(FIXED_OUT_DIR/f"{FIXED_SENSOR}_region_replay_origin_filter_removed.txt",post_removed)

    print("Fixed main-lidar replay:")
    print(f"   input:{FIXED_INPUT}")
    print(f"   compare:{FIXED_COMPARE}")
    print(f"   out_dir:{FIXED_OUT_DIR}")
    print("Relay stats:")

    for key,value  in stats.items():
        print(f"  {key}: {value}")
    print(f" rect_extent(min_x,max_x,min_y,max_y) :{rack_rect_extents(para)}")
    print(f"  filtered_txt: {FIXED_OUT_DIR / f'{FIXED_SENSOR}_region_replay_filtered.txt'}")
    print(f"  removed_txt: {FIXED_OUT_DIR / f'{FIXED_SENSOR}_region_replay_removed.txt'}")
    print(f"  origin_filter_txt: {FIXED_OUT_DIR / f'{FIXED_SENSOR}_region_replay_origin_filter.txt'}")

    expected = read_points(FIXED_COMPARE)
    if len(expected) == len(filtered):
        expected_zero = np.linalg.norm(expected[:, :2], axis=1) <= 1e-6
        replay_zero = np.linalg.norm(filtered[:, :2], axis=1) <= 1e-6
        print("Compare against recorded rack_filter:")
        print(f"  same_zero_mask: {int(np.count_nonzero(expected_zero == replay_zero))}/{len(filtered)}")
        print(f"  expected_zero: {int(np.count_nonzero(expected_zero))}")
        print(f"  replay_zero: {int(np.count_nonzero(replay_zero))}")
        print(f"  mismatch_zero_mask: {int(np.count_nonzero(expected_zero != replay_zero))}")
    else:
        print(f"Compare skipped: size differs, expected={len(expected)}, replay={len(filtered)}")

        cpp_origin_filter = read_points(
        DATA_DIR / "2026-7-2-13-48-59.182.locate_lidar_main_origin_filter.txt"
    )
    if len(cpp_origin_filter) == len(origin_filter):
        expected_zero = np.linalg.norm(cpp_origin_filter[:, :2], axis=1) <= 1e-6
        replay_zero = np.linalg.norm(origin_filter[:, :2], axis=1) <= 1e-6
        expected_100 = np.isclose(np.linalg.norm(cpp_origin_filter[:, :2], axis=1), 100.0, atol=1e-3)
        replay_100 = np.isclose(np.linalg.norm(origin_filter[:, :2], axis=1), 100.0, atol=1e-3)
        print("Compare against recorded origin_filter:")
        print(f"  same_zero_mask: {int(np.count_nonzero(expected_zero == replay_zero))}/{len(origin_filter)}")
        print(f"  expected_zero: {int(np.count_nonzero(expected_zero))}")
        print(f"  replay_zero: {int(np.count_nonzero(replay_zero))}")
        print(f"  mismatch_zero_mask: {int(np.count_nonzero(expected_zero != replay_zero))}")
        print(f"  same_100_mask: {int(np.count_nonzero(expected_100 == replay_100))}/{len(origin_filter)}")
        print(f"  expected_100: {int(np.count_nonzero(expected_100))}")
        print(f"  replay_100: {int(np.count_nonzero(replay_100))}")
        print(f"  mismatch_100_mask: {int(np.count_nonzero(expected_100 != replay_100))}")
    else:
        print(
            f"Origin-filter compare skipped: size differs, expected={len(cpp_origin_filter)}, "
            f"replay={len(origin_filter)}"
        )

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

    if True:
        visualize(origin,origin_filter,post_removed,install,para)
    return 0




if __name__ == "__main__":
    main()
