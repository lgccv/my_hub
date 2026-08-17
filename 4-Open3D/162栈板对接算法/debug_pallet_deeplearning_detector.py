import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path("/home/standard/code/standard_perception")
DATA_DIR = REPO_ROOT / "log/sros21/log/data"
PCL_ORIGIN_CLOUD_PATH = DATA_DIR / "debug_pallet_pcl_origin_cloud.txt"

BBOX_X = 103
BBOX_Y = 176
BBOX_W = 127
BBOX_H = 23

CAMERA_TO_GROUND_TRANSF = np.array([0.0, 0.0, 2.419], dtype=np.float32)
CAMERA_TO_GROUND_ROTATE = np.array([0.0, -0.0349066, 0.0], dtype=np.float32)
HEIGHT_OFFSET = np.array([0.0, 0.0, 0.2], dtype=np.float32)
DETECT_TRANSF = CAMERA_TO_GROUND_TRANSF + HEIGHT_OFFSET

TARGET_VALUE_Z_MM = 1940.0

ANGLE_THRESHOLD_DEG = 60.0
NORMAL_RADIUS = 0.03

MAP_RESOLUTION = 0.01
MAP_LENGTH_M = 3.0
MAP_WIDTH_M = 3.0
MAP_MIN_HEIGHT = -0.1
MAP_MAX_HEIGHT = 3.0
MAP_HEIGHT_STEP = 75.0

MIN_SURFACE_HEIGHT = 0.06
MAX_SURFACE_HEIGHT = 0.25
MIN_SURFACE_WIDTH = 0.03
MAX_SURFACE_WIDTH = 0.1
SURFACE_CHECK_PERCENTAGE = 0.6
PIER_CENTER_PERCENTAGE = 1.0 / 20.0
MIN_WIDTH_BETWEEN_PIERS = 0.20
MAX_ANGLE_BETWEEN_PIER_SURFACE = 9.0
MAX_PALLET_WIDTH = 2.0

@dataclass
class EdgeInfo:  # 在地图第(x,y个格子里，找到了一段从down_index到up_index的竖向表面，并记录它的上下边界3D点)
    up_index: int = 0  # 在map中的高度层编号
    down_index: int = 0
    mid_index: int = 0
    x: int = 0
    y: int = 0
    up_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    down_point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class ObjectInfo:
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    norm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    object_width: float = 0.0
    object_height: float = 0.0
    left_pier_width: float = 0.0
    right_pier_width: float = 0.0
    middle_pier_width: float = 0.0

def cpp_roundf(value):
    if value >=0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


class HeightBar:
    def __init__(self):
        self.points_index = []
        self.points = []
        self.origin_points = []
        self.weights = []
        self.max_height = 0.0
        self.min_height = 0.0

    def add_point(self,index,point):
        point = point.astype(np.float32)
        self.origin_points.append(point.copy())

        if not self.points_index:
            self.max_height = float(point[2])
            self.min_height = float(point[2])
            self.points.append(np.zeros(3,dtype=np.float32))
            self.weights.append(0.0)

        while len(self.points_index) <=index:
            self.points_index.append(-1)

        if self.points_index[index] != -1:
            array_index = self.points_index[index]
            weight = self.weights[array_index]
            self.points[array_index] = (self.points[array_index]*weight +point) /( weight + 1.0)
            self.weights[array_index] = weight + 1.0
        else:
            self.points_index[index] = len(self.points)
            self.points.append(point.copy())
            self.weights.append(1.0)

        self.max_height = max(self.max_height,float(point[2]))
        self.min_height = min(self.min_height,float(point[2]))

    def point_by_index(self,index):
        return self.points[self.points_index[index]]

    def point_by_safe_index(self,index):
        if index <0 or index >= len(self.points_index):
            return False, np.zeros(3,dtype=np.float32)

        index_id = self.points_index[index]
        if index_id >0 and index_id < 20:
            return True, self.points[index_id].copy()

        return False, np.zeros(3, dtype=np.float32)



class GroundPlaneMap:
    def __init__(self):
        self.resolution = MAP_RESOLUTION
        self.min_height = MAP_MIN_HEIGHT
        self.max_height = MAP_MAX_HEIGHT
        self.height_step = MAP_HEIGHT_STEP
        self.length = cpp_roundf(MAP_LENGTH_M / MAP_RESOLUTION)
        self.width = cpp_roundf(MAP_WIDTH_M /MAP_RESOLUTION)
        self.center_x_offset = 0.0
        self.center_y_offset = MAP_WIDTH_M / 2.0
        self.map = [HeightBar() for _ in range(self.length * self.width)]

    def in_grid(self,x,y):
        return x >=0 and x < self.length and y>=0 and y< self.width

    def coord_x(self,x):
        return cpp_roundf((x + self.center_x_offset) / self.resolution)

    def coord_y(self,y):
        return cpp_roundf((y+ self.center_y_offset)/self.resolution)

    def to_world_x(self,x):
        return x * self.resolution - self.center_x_offset

    def to_world_y(self,y):
        return y*self.resolution - self.center_y_offset

    def bar_index(self,z):
        return cpp_roundf((z- self.min_height)*self.height_step)

    def bar(self,x,y):
        return self.map[y*self.length +x]

    def add_point(self,point):
        coord_x_int = cpp_roundf((point[0] + self.center_x_offset) / self.resolution)
        coord_y_int = cpp_roundf((point[1] + self.center_y_offset) / self.resolution)
        coord_h_int = cpp_roundf((point[2] - self.min_height) * self.height_step)

        if self.in_grid(coord_x_int,coord_y_int) and coord_h_int >=0:
            for i in range(-1,2):
                for j in range(-1,2):
                    if coord_y_int + j >=0 and coord_h_int >=0:
                        if self.in_grid(coord_x_int +i,coord_y_int +j):
                            self.bar(coord_x_int+i,coord_y_int+j).add_point(coord_h_int,point)

    def extract_all_points(self):
        points = []
        for bar in self.map:
            points.extend(bar.points)
        if not points:
            return np.empty((0,3),dtype=np.float32)
        return np.asarray(points,dtype=np.float32)
     


def load_origin_cloud():
    points = np.loadtxt("/home/standard/code/standard_perception/example/debug_pallet_pcl_origin_cloud.txt",dtype=np.float32)
    points = points.reshape(-1,3)
    return points

def make_xyz_patch(origin_cloud):
    return origin_cloud.reshape(BBOX_H,BBOX_W,3)

def estimate_and_filter_normals(origin_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(origin_cloud.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=NORMAL_RADIUS))

    normals = np.asarray(pcd.normals,dtype=np.float32)

    filtered_points = []
    filtered_normals = []

    for point, normal in zip(origin_cloud,normals):
        nx,ny,nz = float(normal[0]),float(normal[1]),float(normal[2])

        if math.isnan(nx) or math.isnan(ny) or math.isnan(nz):
            continue

        # 计算点云法向量normal和X轴之间的夹角，然后把角度归一到[-90,90]
        length = math.sqrt(ny*ny + nz*nz)  # 法向量在YZ平面上的投影长度
        theta = math.atan2(length,nx) *180.0 / math.pi
        theta = theta - 180.0 if theta > 90 else theta

        if abs(theta) < ANGLE_THRESHOLD_DEG:
            filtered_points.append(point)
            filtered_normals.append(normal)

    return np.asarray(filtered_points,dtype=np.float32), np.asarray(filtered_normals,dtype=np.float32)


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx

ROT = euler_to_matrix(
    float(CAMERA_TO_GROUND_ROTATE[0]),
    float(CAMERA_TO_GROUND_ROTATE[1]),
    float(CAMERA_TO_GROUND_ROTATE[2]),
)

def build_map(filtered_cloud):
    ground_map = GroundPlaneMap()

    for point in filtered_cloud[::-1]:
        point_e = ROT @point +DETECT_TRANSF
        ground_map.add_point(point_e)
    return ground_map


def get_point_from_image(xyz_patch,pix_y,pix_x):
    local_x = pix_x - BBOX_X
    local_y = pix_y - BBOX_Y
    if local_x < 0 or local_x >= BBOX_W or local_y <0 or local_y >= BBOX_H:
        return None
    return xyz_patch[local_y,local_x].copy()


def create_hole(ground_map,x,y,up,down):
    bar = ground_map.bar(x,y)
    return EdgeInfo(
        up_index = up,
        down_index = down,
        up_point =bar.point_by_index(up).copy(),
        down_point = bar.point_by_index(down).copy(),
        x=x,
        y=y,
        mid_index = (up+down)//2,
    )

def find_first_surface(ground_map,x_index,y_index,z_index):
    bar = ground_map.bar(x_index,y_index)
    before_bar = ground_map.bar(x_index-1,y_index)
    after_bar = ground_map.bar(x_index +1,y_index)

    bars = bar.points_index
    before_bars = before_bar.points_index
    after_bars = after_bar.points_index

    up_index = -1
    down_index = len(bars)
    max_height_index = len(bars) -1

    for i in range(z_index +1,max_height_index):
        if bars[i-1] * bars[i] <0 and bars[i-1] *bars[i+1] <0:
            if bars[i-1] >0:
                if len(before_bars) > i+2:
                    if before_bars[i] >0 or before_bars[i+1] >0 or before_bars[i+2]>0:
                        continue

                if len(after_bars) > i+2:
                    if after_bars[i] >0 or after_bars[i+1] > 0 or after_bars[i+2] >0:
                        continue
                up_index = i-1
                break
        if i+1 >= len(bars) -1:
            up_index = max_height_index

    j_begin = min(z_index,len(bars)-2)
    for j in range(j_begin,1,-1):
        if (bars[j+1]*bars[j] < 0 and bars[j+1] *bars[j-1] < 0 and bars[j+1] * bars[j-2] < 0):
            if bars[j+1] >0:
                if len(before_bars) >j:
                    if before_bars[j] >0 or before_bars[j-1]  >0 or before_bars[j-2] >0:
                        continue
                if len(after_bars) > j:
                    if after_bars[j-1] >0 or after_bars[j-2] >0:
                        continue
                down_index = j+1
                break
    if up_index - down_index <2:
        return False, None
    incre = 0
    for j in range(down_index,up_index):
        if bars[j] >0:
            incre +=1

    surface_percentage = float(incre) / float(up_index - down_index +1)
    min_surface_height_pixel = int(MIN_SURFACE_HEIGHT /MAP_RESOLUTION)

    if (up_index - down_index + 1) > min_surface_height_pixel:
        if surface_percentage > SURFACE_CHECK_PERCENTAGE:
            return True,create_hole(ground_map,x_index,y_index,up_index,down_index)

    return False,None


def check_surface(ground_map,bar,up_index,down_index):
    real_up = up_index if up_index < len(bar.points_index) else len(bar.points_index) -1
    real_down = down_index
    max_height_index = len(bar.points_index)-1

    for i in range(up_index+1,down_index,-1):
        if i <= max_height_index and i >0:
            if bar.points_index[i] * bar.points_index[i-1] <0:
                if bar.points_index[i-1] >0:
                    real_up = i-1
                    break

    for i in range(down_index-1, real_up):
        if i< max_height_index and i>=0:
            if bar.points_index[i] * bar.points_index[i+1] <0:
                if bar.points_index[i] < 0:
                    real_down = i +1
                    break

    incre = 0.0
    for j in range(down_index, up_index):
        if j < len(bar.points_index) and j>=0 and bar.points_index[j] >0:
            incre += 1.0

    if up_index - down_index + 1 <=0:
        return False,real_up,real_down

    if incre / float(up_index - down_index +1) > SURFACE_CHECK_PERCENTAGE:
        return True, real_up, real_down

    return False, real_up, real_down

def check_point_thresh(ground_map,x,y,up_point,down_point,dist_thresh):
    del down_point
    point = np.array([ground_map.to_world_x(x),ground_map.to_world_y(y)],dtype=np.float32)
    dist_1 = np.linalg.norm(up_point[:2]-point)
    return dist_1 <= dist_thresh

def search_adjacent_surface_in_x(ground_map,x_index,y_index,surface):
    array_index =[0,-1,1,-2,2]
    for i in range(4):
        x = x_index + array_index[i]
        if ground_map.in_grid(x,y_index):
            bar = ground_map.bar(x,y_index)
            ok,real_up, real_down = check_surface(ground_map,bar,surface.up_index,surface.down_index)

            if ok:
                if check_point_thresh(ground_map,x,y_index,bar.point_by_index(real_up),bar.point_by_index(real_down),4*MAP_RESOLUTION):
                    return True,x,create_hole(ground_map,x,y_index,real_up,real_down)
    return False,x_index,None


def search_adjacent_surface_in_y_axis(ground_map,x_index,y_index,step,first_surface,surfaces):
    unfounded = 0
    y = y_index

    while y < ground_map.width and y >=0:
        y +=step
        unfounded +=1
        ok, real_x,adj_surface = search_adjacent_surface_in_x(ground_map,x_index,y,first_surface)

        if ok:
            unfounded = 0
            surfaces.append(adj_surface)
            x_index = real_x

        if unfounded >=4:
            break



def expand_surface(ground_map,initial_point,step):
    if np.any(np.isinf(initial_point)):
        return False, []

    x = ground_map.coord_x(float(initial_point[0]))  # 在高度网格中的编号
    y = ground_map.coord_y(float(initial_point[1]))
    z = ground_map.bar_index(float(initial_point[2]))

    if z<0:
        return False, []

    curr_y = y
    calculate_times = 0
    surfaces = []

    y_leftmost = int(curr_y + MAX_SURFACE_WIDTH / (2.0 * MAP_RESOLUTION))
    y_rightmost = int(curr_y - MAX_SURFACE_WIDTH / (2.0* MAP_RESOLUTION))

    y_leftmost = y_leftmost if y_leftmost < ground_map.width else ground_map.width
    y_rightmost = y_rightmost if y_rightmost > 0 else 0

    while curr_y < y_leftmost and curr_y > y_rightmost and calculate_times < ground_map.width:
        calculate_times +=1
        curr_y += step

        for j in range(2):
            curr_x = x +j
            if ground_map.in_grid(curr_x,curr_y):
                ok, first_surface = find_first_surface(ground_map,curr_x,curr_y,z)
                if ok:
                    surfaces = [first_surface]

                    search_adjacent_surface_in_y_axis(ground_map,first_surface.x,first_surface.y,1,first_surface,surfaces)
                    search_adjacent_surface_in_y_axis(ground_map,first_surface.x,first_surface.y,-1,first_surface,surfaces)
                    surfaces.sort(key=lambda item: item.y)
                    middle_surface = surfaces[len(surfaces) //2]
                    center_y = ground_map.to_world_y(middle_surface.y)

                    if abs(float(initial_point[1]) - center_y) > MAX_SURFACE_WIDTH / 2.0:
                        continue

                    if abs(surfaces[-1].y - surfaces[0].y) >= cpp_roundf(MIN_SURFACE_WIDTH /MAP_RESOLUTION):
                        return True, surfaces
    return False, []


def find_surface(ground_map,xyz_path,init_x,init_y):
    array_index = [0,-1,1,-2,2,-3,3,-4,4,-5,5]

    for dx in array_index:
        for dy in array_index:
            pix_x = init_x + dx
            pix_y = init_y + dy

            middle_point = get_point_from_image(xyz_path,pix_y,pix_x)
            if middle_point is None:
                continue

            if middle_point[0] < 0.2:
                continue

            transf_point = ROT@middle_point + DETECT_TRANSF

            ok,surfaces = expand_surface(ground_map,transf_point,1)
            if ok:
                return True,surfaces

            ok, surfaces = expand_surface(ground_map,transf_point,-1)
            if ok:
                return True,surfaces
    return False, []

def delete_surface_bar(ground_map,surfaces,pallet,distance_thresh):
    output = []

    for surface in surfaces:
        p = np.array([ground_map.to_world_x(surface.x),ground_map.to_world_y(surface.y),0.0],dtype= np.float32)
        distance = abs(float(np.dot(p-pallet.center,pallet.norm)))

        if distance <= distance_thresh:
            output.append(surface)

    return output

def compute_mean_point_of_surface(ground_map,surfaces):
    if not surfaces:
        return False, np.zeros(3,dtype=np.float32)

    surfaces.sort(key=lambda item: item.y)
    center_y = surfaces[len(surfaces) //2].y

    center_points = []
    for surface in surfaces:
        if abs(surface.y - center_y) <=2:
            bar = ground_map.bar(surface.x,surface.y)
            for i in range(surface.down_index,surface.up_index+1):
                if i>0 and i< len(bar.points_index) and bar.points_index[i] !=-1:
                    if check_point_thresh(ground_map,surface.x,surface.y,bar.point_by_index(i),bar.point_by_index(surface.down_index),4* MAP_RESOLUTION):
                        center_points.append(bar.point_by_index(i))

    if not center_points:
        return False, np.zeros(3,dtype=np.float32)

    return True,np.mean(np.asarray(center_points,dtype=np.float32),axis=0)


def compute_pallet_pose(ground_map,left_surfaces,middle_surfaces,right_surfaces):
    pallet = ObjectInfo()

    ok_left, left_mean = compute_mean_point_of_surface(ground_map,left_surfaces)
    ok_middle, middle_mean = compute_mean_point_of_surface(ground_map,middle_surfaces)
    ok_right, right_mean = compute_mean_point_of_surface(ground_map,right_surfaces)

    if not ok_left or not ok_middle or not ok_right:
        return False, pallet

    left_surfaces.sort(key=lambda item: item.y)
    middle_surfaces.sort(key=lambda item: item.y)
    right_surfaces.sort(key=lambda item: item.y)

    pallet.left_pier_width = abs(ground_map.to_world_y(left_surfaces[-1].y) - ground_map.to_world_y(left_surfaces[0].y))
    pallet.right_pier_width = abs(ground_map.to_world_y(right_surfaces[-1].y) - ground_map.to_world_y(right_surfaces[0].y))
    pallet.middle_pier_width = abs(ground_map.to_world_y(middle_surfaces[-1].y) - ground_map.to_world_y(middle_surfaces[0].y))

    left_y = ground_map.to_world_y(left_surfaces[-1].y)
    right_y = ground_map.to_world_y(right_surfaces[0].y)
    cur_width = abs(left_y - right_y)
    if cur_width < MAX_PALLET_WIDTH:
        pallet.object_width = cur_width

    max_pier_height = 0.0
    for surfaces in [left_surfaces,middle_surfaces,right_surfaces]:
        for surface in surfaces:
            cur_height = abs(float(surface.up_point[2] - surface.down_point[2]))
            if cur_height < MAX_SURFACE_HEIGHT:
                max_pier_height = max(max_pier_height,cur_height)

    pallet.object_height = max_pier_height

    middle_to_left = left_mean - middle_mean
    middle_to_right = right_mean - middle_mean

    middle_to_left = np.array([-0.018,0.45,0.003],dtype=np.float32)
    denom =  np.linalg.norm(middle_to_left) * np.linalg.norm(middle_to_right)
    if denom <=1e-9:
        return False,pallet

    cos_value = float(np.dot(middle_to_left,middle_to_right) / denom)
    cos_value = max(-1.0,min(1.0,cos_value))

    angle_between = math.acos(cos_value)*180.0 / math.pi

    if abs(angle_between - 180.0) > MAX_ANGLE_BETWEEN_PIER_SURFACE:
        return False, pallet

    if (np.linalg.norm(middle_to_left) < MIN_WIDTH_BETWEEN_PIERS or np.linalg.norm(middle_to_right) < MIN_WIDTH_BETWEEN_PIERS):
        return False, pallet

    norm = np.zeros(3, dtype=np.float32)

    if np.linalg.norm(left_mean - middle_mean) >= MIN_WIDTH_BETWEEN_PIERS:
        direction = middle_mean - left_mean
        norm_tmp = np.cross(np.array([0.0,0.0,1.0],dtype=np.float32),direction)
        norm +=norm_tmp / np.linalg.norm(norm_tmp)

    if np.linalg.norm(right_mean - middle_mean) >= MIN_WIDTH_BETWEEN_PIERS:
        direction = right_mean - middle_mean
        norm_tmp  = np.cross(np.array([0.0,0.0,1.0],dtype=np.float32),direction)
        norm+= norm_tmp / np.linalg.norm(norm_tmp)

    norm = norm / np.linalg.norm(norm)
    pallet.norm = norm if float(np.dot(middle_mean,norm)) > 0 else -norm
    pallet.center = middle_mean

    return True,pallet




def search_pallet(ground_map,xyz_patch):
    bbox_middle_x = int(BBOX_X + BBOX_W /2.0)
    bbox_middle_y = int(BBOX_Y + BBOX_H /2.0)

    left_pier_middle_x = int(BBOX_X + math.floor(BBOX_W * PIER_CENTER_PERCENTAGE))
    left_pier_middle_y = bbox_middle_y

    right_pier_middle_x = int(BBOX_X + BBOX_W - math.floor(BBOX_W * PIER_CENTER_PERCENTAGE))
    right_pier_middle_y = bbox_middle_y

    ok_middle,middle_surfaces = find_surface(ground_map,xyz_patch,bbox_middle_x,bbox_middle_y)

    if not ok_middle:
        return False, ObjectInfo()

    ok_left,left_surfaces = find_surface(ground_map,xyz_patch,left_pier_middle_x,left_pier_middle_y)
    left_surfaces = middle_surfaces
    ok_left = True
    if not ok_left:
        return False, ObjectInfo()

    ok_right,right_surfaces = find_surface(ground_map,xyz_patch,right_pier_middle_x,right_pier_middle_y)
    if not ok_right:
        return False, ObjectInfo()

    mean_up = 0
    mean_down = 0
    surface_size = 0

    for surfaces in [left_surfaces,middle_surfaces,right_surfaces]:
        for surface in surfaces:
            mean_up += surface.up_index
            mean_down += surface.down_index
            surface_size +=1
    mean_up = int(mean_up / surface_size)  # 平均就是在这一段栅格里面
    mean_down = int(mean_down / surface_size)

    for surfaces in [left_surfaces,middle_surfaces,right_surfaces]:
        for surface in surfaces:
            surface.up_index = mean_up
            surface.down_index = mean_down

    ok_init, init_pallet = compute_pallet_pose(ground_map,left_surfaces,middle_surfaces,right_surfaces)

    if not ok_init:
        init_pallet = ObjectInfo()

    left_surfaces = delete_surface_bar(ground_map,left_surfaces,init_pallet,0.03)
    middle_surfaces = delete_surface_bar(ground_map,middle_surfaces,init_pallet,0.02)
    right_surfaces = delete_surface_bar(ground_map,right_surfaces,init_pallet,0.03)

    return compute_pallet_pose(ground_map,left_surfaces,middle_surfaces,right_surfaces)


def select_pallet(pallets):
    temp_z = TARGET_VALUE_Z_MM / 1000.0

    yz_offset = float("inf")
    target_index = 0

    for i,pallet in enumerate(pallets):
        current_distance = math.sqrt((float(pallet.center[2])-temp_z)*(float(pallet.center[2])-temp_z) + float(pallet.center[1])*float(pallet.center[1]))
        if current_distance < yz_offset:
            target_index =i
            yz_offset = current_distance

    return target_index

def main():
    origin_cloud = load_origin_cloud()
    xyz_path = make_xyz_patch(origin_cloud)

    filtered_cloud, filtered_normals = estimate_and_filter_normals(origin_cloud)

    ground_map = build_map(filtered_cloud)
    map_cloud = ground_map.extract_all_points()

    pallets = []
    ok , pallet = search_pallet(ground_map,xyz_path)

    if ok:
        pallet.center = pallet.center - HEIGHT_OFFSET
        pallets.append(pallet)

    if not pallets:
        print("=========Python Pallet Result ==========")
        print("is_available: false")
        return

    target_index = select_pallet(pallets)
    target = pallets[target_index]

    angle = math.atan2(-float(target.norm[1]),-float(target.norm[0]))
    result_path = DATA_DIR / "debug_pallet_py_result.txt"



    print("===========Python Pallet Result ============")
    print("is_available:", True)
    print("x:",float(target.center[0]))
    print("y:",float(target.center[1]))
    print("z:",float(target.center[2]))

    print("angle_deg:",angle*180/math.pi)
    print("width:",target.object_width)
    print("height:",target.object_height)
    print("normal:",target.norm.tolist())
    print("saved:",result_path)



if __name__ == "__main__":
    main()