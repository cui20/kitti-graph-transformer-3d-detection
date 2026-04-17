import random
import numpy as np
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors

Points = namedtuple('Points', ['xyz', 'attr'])

# -------------------------- 点云下采样
def downsample_by_average_voxel(points, voxel_size):
    """平均体素下采样,保留点云几何信息"""
    xmax, ymax, zmax = np.amax(points.xyz, axis=0)
    xmin, ymin, zmin = np.amin(points.xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_idx = (points.xyz - xyz_offset) // voxel_size
    xyz_idx = xyz_idx.astype(np.int32)
    dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
    keys = xyz_idx[:, 0] + xyz_idx[:, 1]*dim_x + xyz_idx[:, 2]*dim_y*dim_x
    order = np.argsort(keys)
    keys = keys[order]
    points_xyz = points.xyz[order]
    unique_keys, lens = np.unique(keys, return_counts=True)
    indices = np.hstack([[0], lens[:-1]]).cumsum()
    downsampled_xyz = np.add.reduceat(points_xyz, indices, axis=0)/lens[:, np.newaxis]

    include_attr = points.attr is not None
    if include_attr:
        attr = points.attr[order]
        downsampled_attr = np.add.reduceat(attr, indices, axis=0)/lens[:, np.newaxis]
        return Points(xyz=downsampled_xyz, attr=downsampled_attr)
    return Points(xyz=downsampled_xyz, attr=None)

# -------------------------- 3D框处理
def box3d_to_cam_points(label, expend_factor=(1.0, 1.0, 1.0)):
    """将3D标注框转换为相机坐标系下的8个角点"""
    yaw = label['yaw'] if 'yaw' in label else label['rotation_y'] if 'rotation_y' in label else 0.0
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    h = label['height']
    delta_h = h*(expend_factor[0]-1)
    w = label['width']*expend_factor[1]
    l = label['length']*expend_factor[2]
    corners = np.array([
        [ l/2, delta_h/2, w/2], [ l/2, delta_h/2, -w/2],
        [-l/2, delta_h/2, -w/2], [-l/2, delta_h/2, w/2],
        [ l/2, -h-delta_h/2, w/2], [ l/2, -h-delta_h/2, -w/2],
        [-l/2, -h-delta_h/2, -w/2], [-l/2, -h-delta_h/2, w/2]
    ])
    r_corners = corners.dot(np.transpose(R))
    tx = label['x3d'] if 'x3d' in label else label['location'][0] if 'location' in label else 0.0
    ty = label['y3d'] if 'y3d' in label else label['location'][1] if 'location' in label else 0.0
    tz = label['z3d'] if 'z3d' in label else label['location'][2] if 'location' in label else 0.0
    cam_points_xyz = r_corners + np.array([tx, ty, tz])
    return Points(xyz=cam_points_xyz, attr=None)

def box3d_to_normals(label, expend_factor=(1.0, 1.0, 1.0)):
    """计算3D框的法向量,用于点筛选"""
    box3d_points = box3d_to_cam_points(label, expend_factor)
    box3d_points_xyz = box3d_points.xyz
    wx = box3d_points_xyz[[0], :] - box3d_points_xyz[[4], :]
    lx = np.matmul(wx, box3d_points_xyz[4, :])
    ux = np.matmul(wx, box3d_points_xyz[0, :])
    wy = box3d_points_xyz[[0], :] - box3d_points_xyz[[1], :]
    uy = np.matmul(wy, box3d_points_xyz[0, :])
    ly = np.matmul(wy, box3d_points_xyz[1, :])
    wz = box3d_points_xyz[[0], :] - box3d_points_xyz[[3], :]
    lz = np.matmul(wz, box3d_points_xyz[3, :])
    uz = np.matmul(wz, box3d_points_xyz[0, :])
    return (np.concatenate([wx, wy, wz], axis=0),
            np.concatenate([lx, ly, lz]),
            np.concatenate([ux, uy, uz]))

def sel_xyz_in_box3d(label, xyz, expend_factor=(1.0, 1.0, 1.0)):
    """筛选3D框内的点,返回掩码"""
    normals, lower, upper = box3d_to_normals(label, expend_factor)
    projected = np.matmul(xyz, np.transpose(normals))
    points_in_x = np.logical_and(projected[:, 0] > lower[0], projected[:, 0] < upper[0])
    points_in_y = np.logical_and(projected[:, 1] > lower[1], projected[:, 1] < upper[1])
    points_in_z = np.logical_and(projected[:, 2] > lower[2], projected[:, 2] < upper[2])
    mask = np.logical_and.reduce((points_in_x, points_in_y, points_in_z))
    return mask

# -------------------------- 自适应KNN图构建
def build_adaptive_knn_graph(points, k_min=5, k_max=20, density_radius=1.0):
    """构建自适应K近邻图(替代Point-GNN的固定半径图)"""
    xyz = points.xyz
    num_points = xyz.shape[0]

    # 1. 计算每个点的局部密度(固定半径内的点数)
    nn_density = NearestNeighbors(radius=density_radius)
    nn_density.fit(xyz)
    density = nn_density.radius_neighbors(xyz, return_distance=False)
    local_density = np.array([len(neighbors) for neighbors in density])

    # 2. 自适应分配K值:密度越大,K越小
    density_normalized = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)
    k_per_point = k_max - (k_max - k_min) * density_normalized
    k_per_point = k_per_point.astype(np.int32)
    k_per_point = np.clip(k_per_point, k_min, k_max)

    # 3. 全局K近邻搜索(取最大K,后续按每个点的K裁剪)
    nn_knn = NearestNeighbors(n_neighbors=k_max)
    nn_knn.fit(xyz)
    _, knn_idx = nn_knn.kneighbors(xyz)

    # 4. 按每个点的自适应K值裁剪,构建边索引
    edge_index = []
    for i in range(num_points):
        k = k_per_point[i]
        neighbors = knn_idx[i, :k]
        for j in neighbors:
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = np.array(edge_index).T
    return edge_index, knn_idx, k_per_point
