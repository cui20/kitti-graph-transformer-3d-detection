import os
from os.path import isfile, join
import numpy as np
import cv2
from collections import namedtuple

from .utils import downsample_by_average_voxel, sel_xyz_in_box3d

Points = namedtuple('Points', ['xyz', 'attr'])

# -------------------------- 核心KITTI数据集类
class KittiDataset:
    def __init__(self, image_dir, point_dir, calib_dir, label_dir,
                 index_filename=None, is_training=True, difficulty=-1, num_classes=4):
        self._image_dir = image_dir
        self._point_dir = point_dir
        self._calib_dir = calib_dir
        self._label_dir = label_dir
        self._index_filename = index_filename

        if index_filename:
            self._file_list = self._read_index_file(index_filename)
        else:
            self._file_list = self._get_file_list(self._image_dir)

        self._verify_file_list(is_training)
        self._is_training = is_training
        self.num_classes = num_classes
        self.difficulty = difficulty

    def __len__(self):
        return len(self._file_list)

    @property
    def num_files(self):
        return len(self._file_list)

    def _read_index_file(self, index_filename):
        file_list = []
        with open(index_filename, 'r') as f:
            for line in f:
                file_list.append(line.rstrip('\n').split('.')[0])
        return file_list

    def _get_file_list(self, image_dir):
        file_list = [f.split('.')[0] for f in os.listdir(image_dir) if isfile(join(image_dir, f))]
        file_list.sort()
        return file_list

    def _verify_file_list(self, is_training):
        for f in self._file_list:
            assert isfile(join(self._image_dir, f)+'.png'), f"图像文件不存在: {f}"
            assert isfile(join(self._point_dir, f)+'.bin'), f"点云文件不存在: {f}"
            assert isfile(join(self._calib_dir, f)+'.txt'), f"标定文件不存在: {f}"
            if is_training:
                assert isfile(join(self._label_dir, f)+'.txt'), f"标签文件不存在: {f}"

    def get_calib(self, frame_idx):
        """加载标定矩阵,完成坐标系转换预计算"""
        calib_file = join(self._calib_dir, self._file_list[frame_idx])+'.txt'
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                fields = line.split(' ')
                matrix_name = fields[0].rstrip(':')
                matrix = np.array(fields[1:], dtype=np.float32)
                calib[matrix_name] = matrix

        # 矩阵维度调整与坐标系转换预计算
        calib['P2'] = calib['P2'].reshape(3, 4)
        calib['R0_rect'] = calib['R0_rect'].reshape(3,3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3,4)

        R0_rect = np.eye(4)
        R0_rect[:3, :3] = calib['R0_rect']
        calib['velo_to_rect'] = np.vstack([calib['Tr_velo_to_cam'], [0,0,0,1]])
        calib['cam_to_image'] = np.hstack([calib['P2'][:, 0:3], [[0],[0],[0]]])
        calib['rect_to_cam'] = np.hstack([
            calib['R0_rect'],
            np.matmul(np.linalg.inv(calib['P2'][:, 0:3]), calib['P2'][:, [3]])
        ])
        calib['rect_to_cam'] = np.vstack([calib['rect_to_cam'], [0,0,0,1]])
        calib['velo_to_cam'] = np.matmul(calib['rect_to_cam'], calib['velo_to_rect'])
        calib['cam_to_velo'] = np.linalg.inv(calib['velo_to_cam'])
        calib['velo_to_image'] = np.matmul(calib['cam_to_image'], calib['velo_to_cam'])

        return calib

    def get_filename(self, frame_idx):
        return self._file_list[frame_idx]

    def get_velo_points(self, frame_idx, xyz_range=None):
        """加载原始激光雷达点云"""
        point_file = join(self._point_dir, self._file_list[frame_idx])+'.bin'
        velo_data = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
        velo_points = velo_data[:, :3]
        reflections = velo_data[:, [3]]

        if xyz_range is not None:
            x_range, y_range, z_range = xyz_range
            mask = (velo_points[:, 0] > x_range[0]) & (velo_points[:, 0] < x_range[1])
            mask &= (velo_points[:, 1] > y_range[0]) & (velo_points[:, 1] < y_range[1])
            mask &= (velo_points[:, 2] > z_range[0]) & (velo_points[:, 2] < z_range[1])
            return Points(xyz=velo_points[mask], attr=reflections[mask])

        return Points(xyz=velo_points, attr=reflections)

    def velo_points_to_cam(self, points, calib):
        """激光雷达坐标系 ￫ 相机坐标系"""
        cam_xyz = np.matmul(points.xyz, np.transpose(calib['velo_to_cam'])[:3,:3].astype(np.float32))
        cam_xyz += np.transpose(calib['velo_to_cam'])[[3], :3].astype(np.float32)
        return Points(xyz=cam_xyz, attr=points.attr)

    def cam_points_to_image(self, points, calib):
        """相机坐标系 ￫ 图像坐标系"""
        cam_xyz1 = np.hstack([points.xyz, np.ones([points.xyz.shape[0], 1])])
        img_points_xyz = np.matmul(cam_xyz1, np.transpose(calib['cam_to_image']))
        img_points_xy1 = img_points_xyz / img_points_xyz[:, [2]]
        return Points(img_points_xy1, points.attr)

    def get_cam_points_in_image(self, frame_idx, downsample_voxel_size=None, calib=None, xyz_range=None):
        """仅保留相机视野内的点云(过滤无效点)"""
        if calib is None:
            calib = self.get_calib(frame_idx)
        velo_points = self.get_velo_points(frame_idx, xyz_range=xyz_range)
        cam_points = self.velo_points_to_cam(velo_points, calib)

        if downsample_voxel_size is not None:
            cam_points = downsample_by_average_voxel(cam_points, downsample_voxel_size)

        image = self.get_image(frame_idx)
        height, width = image.shape[:2]

        # 仅保留相机前方、视野内的点
        front_mask = cam_points.xyz[:, 2] > 0.1
        front_points = Points(cam_points.xyz[front_mask], cam_points.attr[front_mask])
        img_points = self.cam_points_to_image(front_points, calib)
        in_image_mask = (img_points.xyz[:,0]>0) & (img_points.xyz[:,0]<width) & \
                        (img_points.xyz[:,1]>0) & (img_points.xyz[:,1]<height)
        return Points(xyz=front_points.xyz[in_image_mask], attr=front_points.attr[in_image_mask])

    def get_image(self, frame_idx):
        """加载相机图像"""
        image_file = join(self._image_dir, self._file_list[frame_idx])+'.png'
        return cv2.imread(image_file)

    def get_label(self, frame_idx):
        """加载3D检测标注,按难度过滤"""
        MIN_HEIGHT = [40, 25, 25]
        MAX_OCCLUSION = [0, 1, 2]
        MAX_TRUNCATION = [0.15, 0.3, 0.5]

        label_file = join(self._label_dir, self._file_list[frame_idx])+'.txt'
        label_list = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(' ')
                label = {
                    'name': fields[0],
                    'truncation': float(fields[1]),
                    'occlusion': int(fields[2]),
                    'alpha': float(fields[3]),
                    'xmin': float(fields[4]),
                    'ymin': float(fields[5]),
                    'xmax': float(fields[6]),
                    'ymax': float(fields[7]),
                    'height': float(fields[8]),
                    'width': float(fields[9]),
                    'length': float(fields[10]),
                    'x3d': float(fields[11]),
                    'y3d': float(fields[12]),
                    'z3d': float(fields[13]),
                    'yaw': float(fields[14])
                }
                if len(fields) > 15:
                    label['score'] = float(fields[15])

                # 按难度过滤
                if self.difficulty > -1:
                    if label['truncation'] > MAX_TRUNCATION[self.difficulty]:
                        continue
                    if label['occlusion'] > MAX_OCCLUSION[self.difficulty]:
                        continue
                    if (label['ymax'] - label['ymin']) < MIN_HEIGHT[self.difficulty]:
                        continue

                label_list.append(label)
        return label_list

    def assign_car_label_to_points(self, labels, xyz, expend_factor=(1.1, 1.1, 1.1)):
        """给每个点分配汽车类别标签与3D框(4分类:背景、正/侧向汽车、忽略)"""
        assert self.num_classes == 4
        num_points = xyz.shape[0]
        cls_labels = np.zeros((num_points, 1), dtype=np.int64)
        boxes_3d = np.zeros((num_points, 7))
        valid_boxes = np.zeros((num_points, 1), dtype=np.float32)
        label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}

        for label in labels:
            obj_cls = label_map.get(label['name'], 3)
            if obj_cls == 1:
                mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                yaw = label['yaw']

                # 归一化朝向角,区分正/侧向汽车
                while yaw < -0.25*np.pi:
                    yaw += np.pi
                while yaw > 0.75*np.pi:
                    yaw -= np.pi

                if yaw < 0.25*np.pi:
                    cls_labels[mask] = 1
                else:
                    cls_labels[mask] = 2

                # 赋值3D框参数
                boxes_3d[mask] = (label['x3d'], label['y3d'], label['z3d'],
                                  label['length'], label['height'], label['width'], yaw)
                valid_boxes[mask] = 1
            elif obj_cls != 3:
                mask = sel_xyz_in_box3d(label, xyz, expend_factor)
                cls_labels[mask] = obj_cls
                valid_boxes[mask] = 0.0

        return cls_labels, boxes_3d, valid_boxes

# -------------------------- 简化版数据集类(用于推理)
class SimpleKittiDataset:
    def __init__(self, image_dir, point_dir, calib_dir, label_dir):
        self._image_dir = image_dir
        self._point_dir = point_dir
        self._calib_dir = calib_dir
        self._label_dir = label_dir
        self._file_list = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]
        self._file_list.sort()

    def __len__(self):
        return len(self._file_list)

    def get_filename(self, idx):
        return self._file_list[idx]

    def get_calib(self, idx):
        calib_file = os.path.join(self._calib_dir, self._file_list[idx])+'.txt'
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                fields = line.split(' ')
                matrix_name = fields[0].rstrip(':')
                matrix = np.array(fields[1:], dtype=np.float32)
                calib[matrix_name] = matrix

        calib['P2'] = calib['P2'].reshape(3, 4)
        calib['R0_rect'] = calib['R0_rect'].reshape(3,3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3,4)
        calib['velo_to_cam'] = np.vstack([calib['Tr_velo_to_cam'], [0,0,0,1]])
        return calib

    def get_velo_points(self, idx):
        point_file = os.path.join(self._point_dir, self._file_list[idx])+'.bin'
        velo_data = np.fromfile(point_file, dtype=np.float32).reshape(-1, 4)
        return Points(xyz=velo_data[:, :3], attr=velo_data[:, [3]])

    def velo_points_to_cam(self, points, calib):
        cam_xyz = np.matmul(points.xyz, calib['velo_to_cam'][:3,:3].T) + calib['velo_to_cam'][:3,3]
        return Points(xyz=cam_xyz, attr=points.attr)

    def get_cam_points_in_image(self, idx, downsample_voxel_size=0.8, calib=None):
        if calib is None:
            calib = self.get_calib(idx)
        velo_points = self.get_velo_points(idx)
        cam_points = self.velo_points_to_cam(velo_points, calib)

        if downsample_voxel_size:
            from .utils import downsample_by_average_voxel
            cam_points = downsample_by_average_voxel(cam_points, downsample_voxel_size)

        front_mask = cam_points.xyz[:, 2] > 0.1
        return Points(xyz=cam_points.xyz[front_mask], attr=cam_points.attr[front_mask])

    def get_label(self, idx):
        label_file = os.path.join(self._label_dir, self._file_list[idx])+'.txt'
        label_list = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(' ')
                label = {
                    'name': fields[0],
                    'truncation': float(fields[1]),
                    'occlusion': int(fields[2]),
                    'alpha': float(fields[3]),
                    'xmin': float(fields[4]),
                    'ymin': float(fields[5]),
                    'xmax': float(fields[6]),
                    'ymax': float(fields[7]),
                    'height': float(fields[8]),
                    'width': float(fields[9]),
                    'length': float(fields[10]),
                    'x3d': float(fields[11]),
                    'y3d': float(fields[12]),
                    'z3d': float(fields[13]),
                    'yaw': float(fields[14])
                }
                label_list.append(label)
        return label_list
