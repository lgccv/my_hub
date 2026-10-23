import open3d as o3d

from Open3D import *


def filter_by_y_axis_normal(pcd,normal_radius=0.015,max_nn=30,min_y_angle_deg=70.0,max_z_deviation_deg=20.0):
    if pcd is None or len(pcd.points) <3:
        print("法线过滤失败：输入点云为空或点数不足")
        return o3d.geometry.PointCloud()

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius,max_nn=max_nn))

    pcd.normalize_normals()

    normals = np.asarray(pcd.normals,dtype=np.float64)
    normal_lengths = np.linalg.norm(normals,axis=1)

    valid_mask = (
        np.all(np.isfinite(normals),axis=1)
        & np.isfinite(normal_lengths)
        & (normal_lengths > np.finfo(np.float64).eps)
    )

    # 法线正反方向不确定，所以使用绝对值
    cos_to_y = np.zeros(len(normals),dtype=np.float64)
    cos_to_y[valid_mask] = (np.abs(normals[valid_mask,1]) / normal_lengths[valid_mask])
    cos_to_y = np.clip(cos_to_y,0.0,1.0)
    angle_to_y_deg = np.degrees(np.arccos(cos_to_y))

    # 限制法线接近XY平面，避免保留顶部、底部等Z向法线
    max_abs_normal_z = math.sin(math.radians(max_z_deviation_deg))
    normalized_abs_z = np.full(len(normals),np.inf,dtype=np.float64)
    normalized_abs_z[valid_mask] = (np.abs(normals[valid_mask,2]) / normal_lengths[valid_mask])

    keep_mask = (valid_mask & (angle_to_y_deg >= min_y_angle_deg) & (normalized_abs_z <= max_abs_normal_z))

    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        print("法线过滤失败: 没有满足条件的点")
        return o3d.geometry.PointCloud()
    filtered_pcd = pcd.select_by_index(keep_indices.tolist())

    print(
        "法线过滤完成：",
        "原始点数=", len(pcd.points),
        "保留点数=", len(filtered_pcd.points),
        "Y轴最小夹角=", min_y_angle_deg,
        "Z方向最大偏差=", max_z_deviation_deg,
    )

    return filtered_pcd



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


if __name__ == "__main__":

    path2 =r"/Users/jodocls/Downloads/10.10.71.198/202608281723/two_piers_pcl_right_pillar_cloud_4_.txt"


    path = r"/Users/jodocls/Downloads/10.10.71.198/202608281605/two_piers_pcl_left_pillar_cloud_3_.txt"

    point = load_xyz(path=path)

    pcd = points_to_pcd(point)

    thresh_angle = 60


    # 法向量过滤
    cal_estimate_normals(pcd=pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.02))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    filtered_points = []
    for point, normal in zip(points,normals):
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
        if math.isnan(nx) or math.isnan(ny) or math.isnan(nz):
            continue
        length = math.sqrt(nx*nx + nz*nz)
        theta = math.atan2(length,ny) * 180 / math.pi
        theta = theta - 180 if theta > 90 else theta

        if abs(theta) > thresh_angle:
            filtered_points.append(point)

    filter_pcd = points_to_pcd(np.asarray(filtered_points,dtype=np.float64))

    cal_estimate_normals(filter_pcd)

    show_ply(filter_pcd,show_normal=True)

    # 直接采用拟合平面方式过滤
    plane_model,inlier_indices = filter_pcd.segment_plane(distance_threshold = 0.005,ransac_n = 3, num_iterations = 3000)
    ## 提取结果
    a,b,c,d = plane_model
    inliers = filter_pcd.select_by_index(inlier_indices)

    show_ply(inliers,True)



