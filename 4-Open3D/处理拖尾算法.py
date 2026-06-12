# 拖尾过滤原理:
# 由于拖尾的角度一致性，转为(r,a)后，可以通过a的聚类，过滤掉
# 聚类本质是一种分割算法，如果不属于任何一个簇，就是异常点
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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


def plot_cylindrical_pcd(cylindrical_pcd, out_png=None):
    cylindrical_np = np.asarray(cylindrical_pcd.points, dtype=np.float64)

    r = cylindrical_np[:, 0]
    theta = cylindrical_np[:, 1]

    # 如果想让横坐标更直观，可以转成角度制
    theta_deg = np.degrees(theta)

    plt.figure(figsize=(10, 6))
    plt.scatter(theta_deg, r, s=4, c=r, cmap="viridis")
    plt.xlabel("theta (deg)")
    plt.ylabel("radius")
    plt.title("Polar View: angle vs radius")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_png is not None:
        plt.savefig(out_png, dpi=200)
    # plt.show()

# 笛卡尔坐标转为极坐标
def cartesian_to_cylindrical(pcd):
    points = np.asarray(pcd.points, dtype=np.float64)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    # 发射源坐标
    origin_points = [0.44,0,0.165]
    x0 = origin_points[0]
    y0 = origin_points[1]
    z0 = origin_points[2]

    r = np.sqrt((x-x0)**2 + (y-y0)**2+ (z-z0)**2)
    theta = np.degrees(np.arcsin((z-z0) / r))

    cylindrical_np = np.column_stack([r,theta,np.zeros_like(r)])

    cylindrical_pcd = o3d.geometry.PointCloud()
    cylindrical_pcd.points = o3d.utility.Vector3dVector(cylindrical_np)

    indices = np.arange(len(points))
    return cylindrical_pcd, indices


def remove_trailing_by_polar_dbscan(points,cylindrical_pcd,theta_min_deg=35,theta_max_deg=45,theta_scale=1.0,radius_scale=0.2,eps=1.5,min_points=5):
    cylindrical_np = np.asanyarray(cylindrical_pcd.points,dtype=np.float64)
    radius = cylindrical_np[:,0]
    theta = cylindrical_np[:,1]
    

    # 把theta和radius归一到可比较的尺度
    features = np.column_stack([theta/theta_scale,radius/radius_scale,np.zeros_like(radius)])
    feature_pcd = o3d.geometry.PointCloud()
    feature_pcd.points = o3d.utility.Vector3dVector(features)

    labels = np.asarray(feature_pcd.cluster_dbscan(eps=eps,min_points=min_points,print_progress=False))
    remove_mask = np.zeros(len(points),dtype=bool)

    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_theta = theta[cluster_mask]
        cluster_radius = radius[cluster_mask]
        
        theta_median = np.median(cluster_theta)
        theta_min = cluster_theta.min()
        theta_max = cluster_theta.max()
        radius_min = cluster_radius.min()
        radius_max = cluster_radius.max()

        shoule_remove = theta_min_deg <= theta_median <=theta_max_deg

        if shoule_remove:
            remove_mask |=cluster_mask
    
    keep_mask =~remove_mask
    filtered_points = points[keep_mask]

    filtered_points_pcd = o3d.geometry.PointCloud()
    filtered_points_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_points_pcd,keep_mask,remove_mask,labels


def main():
    # 加载点云
    input_path = r"/Users/jodocls/Desktop/code/open3d/project1/2026-5-18-16-49-54.137.110_points_in_sensor.txt"
    point_cloud_np = load_xyz(input_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])

    # 选取ROI区域
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[0.4, -0.5, 0.1],max_bound=[2.0, 0.5, 0.7])
    bbox.color=[0,1,0]
    cropped_pcd = pcd.crop(bbox)
    o3d.visualization.draw_geometries([bbox,cropped_pcd])

    # 对点云进行下采样
    down_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.4)
    o3d.visualization.draw_geometries([down_pcd])

    # 将点云转为极坐标,并记录原始索引
    cylindrical_pcd,original_indices = cartesian_to_cylindrical(pcd)
    filtered_points_pcd,keep_mask,remove_mask,labels = remove_trailing_by_polar_dbscan(
        point_cloud_np[:,:3],
        cylindrical_pcd,
        theta_min_deg=35,
        theta_max_deg=45,
        theta_scale=1.0,
        radius_scale=0.2,
        eps=1.5,
        min_points=5)
    # plot_cylindrical_pcd(filtered_points_pcd,"./result.png")

    # 半径滤波
    clean_pcd, kept_indices = filtered_points_pcd.remove_radius_outlier(  # 含义是：如果一个点在给定半径内邻居数不够，就认为它是离群点。
        nb_points=5,
        radius=0.1)
    
    o3d.visualization.draw_geometries([clean_pcd])

if __name__ == "__main__":
    main()