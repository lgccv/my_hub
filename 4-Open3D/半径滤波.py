import numpy as np
import open3d as o3d

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


def filter_outlier_points_by_radius(points: np.ndarray, radius=0.05, min_neighbors=5):
    """
    points: numpy array, shape = (N, 3)
    radius: 半径搜索距离
    min_neighbors: 最少邻居数，包含自身
    """
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    kept_indices = []

    for i, point in enumerate(pcd.points):
        num_neighbors, indices, distances2 = kdtree.search_radius_vector_3d(
            point,
            radius
        )

        # Open3D 这里包含自身
        if num_neighbors >= min_neighbors:
            kept_indices.append(i)

    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    return points[kept_indices]

def filter_outlier_points_by_radius_open3d(points: np.ndarray, radius=0.05, min_neighbors=5):
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, kept_indices = pcd.remove_radius_outlier(
        nb_points=min_neighbors,
        radius=radius,
    )
    kept_indices = np.asarray(kept_indices, dtype=np.int64)
    return points[kept_indices]



if __name__ == "__main__":
    input_path = r"./data/2026-5-25-9-59-41.144.111_fliter_points_in_base.txt"
    point_cloud_np = load_xyz(input_path)
    filtered_points = filter_outlier_points_by_radius(point_cloud_np,0.05,6)

    save_xyz(filtered_points,"./result.txt")
    print(f"原始点数: {len(point_cloud_np)}")
    print(f"过滤后点数: {len(filtered_points)}")

    # 如果用open3d的代码
    filtered_points = filter_outlier_points_by_radius_open3d(point_cloud_np,0.05,5)
    print(f"原始点数: {len(point_cloud_np)}")
    print(f"过滤后点数: {len(filtered_points)}")