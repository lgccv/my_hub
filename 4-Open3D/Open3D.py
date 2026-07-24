import open3d as o3d
import numpy as np

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
def reduce_roi(pcd,min_bound,max_bound):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[0.4, -0.5, 0.1],max_bound=[2.0, 0.5, 0.7])
    bbox.color=[0,1,0]
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd

# 半径滤波
def radius_filter(pcd):
    clean_pcd, kept_indices = pcd.remove_radius_outlier(  # 含义是：如果一个点在给定半径内邻居数不够，就认为它是离群点。
        nb_points=5,
        radius=0.1)
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

