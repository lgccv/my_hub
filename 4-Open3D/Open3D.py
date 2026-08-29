import open3d as o3d
import numpy as np
import math

# 加载txt文件
def load_xyz(path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3 columns x y z, got shape {points.shape}")
    return points

# 保存txt文件
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


# 点转为pcd
def points_to_pcd(point_cloud_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])
    return pcd

# 展示pcd
def show_ply(pcd:tuple,show_normal = False,show_axis = True,axis_size = 0.1):
    geometries = list(pcd) if isinstance(pcd,(list,tuple)) else [pcd]
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size,
            origin=[0,0,0]
        )
        geometries.append(axis)
    o3d.visualization.draw_geometries(geometries,point_show_normal = show_normal)

# 生成中心点
def gen_circle(center):
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1.0,0.5,1.0])
    return center_sphere


def rotation_matrix_from_vectors(vec1,vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    v = np.cross(vec1,vec2)
    c = np.dot(vec1, vec2)
    if np.isclose(c,1.0):
        return np.eye(3)

    if np.isclose(c,-1.0):
        return np.array([
            [1.0,0.0,0.0],
            [0.0,-1.0,0.0],
            [0.0,0.0,-1.0]
        ])

    vx = np.array([
        [0.0,-v[2],v[1]],
        [v[2],0.0,-v[0]],
        [-v[1],v[0],0.0],
    ])

    R = np.eye(3) +vx +vx@vx * ((1.0-c)/(np.linalg.norm(v))**2)
    return R

# 生成法线方向
def gen_esitimate(center,normal,normal_length=0.1):
    normal = normal / np.linalg.norm(normal)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius = 0.005,
        cone_radius = 0.015,
        cylinder_height = normal_length * 0.8,
        cone_height = normal_length *0.2,
    )
    arrow.paint_uniform_color([1.0,0.0,0.0])

    z_axis = np.array([0.0,0.0,1.0])
    R = rotation_matrix_from_vectors(z_axis,normal)
    arrow.rotate(R,center=np.array([0.0,0.0,0.0]))
    arrow.translate(center)
    return arrow


# 下采样
def down_simple(ply,voxel_size = 0.1):
    down_pcd = ply.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd

# 裁剪ROI区域
def reduce_roi(pcd,x_min,x_max,y_min,y_max,z_min,z_max):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_min, y_min, z_min],max_bound=[x_max, y_max, z_max])
    bbox.color=[0,1,0]
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd

# 半径滤波
def radius_filter(pcd,num=5,radius=0.05):
    clean_pcd, kept_indices = pcd.remove_radius_outlier(  # 含义是：如果一个点在给定半径内邻居数不够，就认为它是离群点。
        nb_points=num,
        radius=radius)
    return clean_pcd,kept_indices

# 计算法线
def cal_estimate_normals(pcd):
    pcd.estimate_normals(search_params = o3d.geometry.KDTreeSearchParamHybrid(radius =0.03,max_nn=30))


# 读取pcd
def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd

# 拟合平面
def fit_plane(pcd,distance_threshold,ransac_n,num_iterations):
    plane_model,inlier_indices = pcd.segment_plane(distance_threshold = distance_threshold,
                                                   ransac_n = ransac_n,
                                                   num_iterations = num_iterations)
    ## 提取结果
    a,b,c,d = plane_model
    inliers = pcd.select_by_index(inlier_indices)
    return a,b,c,d,inliers


# 通过欧拉角构造旋转矩阵
def euler_to_matrix(roll,pitch,yaw):
    cr, sr = math.cos(roll),math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return rz @ ry @ rx


# 通过法线角度过滤点云
def estimate_and_filter_normals(pcd,normal_radius = 0.03,angle_threshold_deg = 60.0):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    filtered_points = []

    for point, normal in zip(points,normals):
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])

        if math.isnan(nx) or math.isnan(ny) or math.isnan(nz):
            continue

        length = math.sqrt(ny*ny + nz*nz)
        theta = math.atan2(length,nx) * 180 / math.pi
        theta = theta - 180 if theta > 90 else theta

        if abs(theta) < angle_threshold_deg:
            filtered_points.append(point)

    if len(filtered_points) == 0:
        return o3d.geometry.PointCloud()

    return points_to_pcd(np.asarray(filtered_points,dtype=np.float64))


# 计算法向量和某个轴的夹角
def angle_of_norm_axis(norm,axis):
    nx,ny,nz = float(norm[0]),float(norm[1]),float(norm[2])
    normal_length = math.sqrt(nx*nx+ny*ny+nz*nz)

    if axis == "x":
        cos_theta = abs(nx) / normal_length
    if axis == "y":
        cos_theta = abs(ny) / normal_length
    if axis == "z":
        cos_theta = abs(nz) / normal_length

    theta = math.acos(cos_theta) * 180 / math.pi
    return theta
