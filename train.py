import os
import random
from collections import namedtuple
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR

from src.data.dataset import KittiDataset
from src.models.graph_transformer import GraphTransformerDetector, ImprovedDetectionLoss

# 点云数据结构定义
Points = namedtuple('Points', ['xyz', 'attr'])

# 数据集路径配置（本地运行请修改为你的路径）
DATASET_ROOT = "/Users/你的用户名/Downloads/kitti"  # Mac用户改这里！
IMAGE_DIR = f"{DATASET_ROOT}/training/image_2"
POINT_DIR = f"{DATASET_ROOT}/training/velodyne"
CALIB_DIR = f"{DATASET_ROOT}/training/calib"
LABEL_DIR = f"{DATASET_ROOT}/training/label_2"
TRAIN_INDEX = None
VAL_INDEX = None

# -------------------------- 带缓存的KITTI数据集
class CachedKittiGraphDataset(Dataset):
    def __init__(self, kitti_dataset, voxel_size=0.8, is_training=True):
        self.kitti_dataset = kitti_dataset
        self.voxel_size = voxel_size
        self.is_training = is_training
        self.num_classes = kitti_dataset.num_classes
        self.cache = {}

    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, idx):
        frame_id = self.kitti_dataset.get_filename(idx)
        if frame_id in self.cache:
            return self.cache[frame_id]

        # 加载点云与标签
        calib = self.kitti_dataset.get_calib(idx)
        points = self.kitti_dataset.get_cam_points_in_image(
            idx, downsample_voxel_size=self.voxel_size, calib=calib
        )
        labels = self.kitti_dataset.get_label(idx)

        # 数据增强(训练时)
        if self.is_training:
            # 随机全局翻转
            if random.random() > 0.5:
                points = Points(xyz=points.xyz * np.array([-1, 1, 1]), attr=points.attr)
                for label in labels:
                    label['x3d'] = -label['x3d']
                    label['yaw'] = -label['yaw']

        # 构建自适应KNN图
        from src.data.utils import build_adaptive_knn_graph
        edge_index, _, _ = build_adaptive_knn_graph(points)

        # 标签分配
        cls_labels, box_labels, valid_mask = self.kitti_dataset.assign_car_label_to_points(
            labels, points.xyz
        )

        # 转换为PyTorch张量
        data = {
            'xyz': torch.from_numpy(points.xyz).float(),
            'attr': torch.from_numpy(points.attr).float(),
            'edge_index': torch.from_numpy(edge_index).long(),
            'cls_labels': torch.from_numpy(cls_labels).long(),
            'box_labels': torch.from_numpy(box_labels).float(),
            'valid_mask': torch.from_numpy(valid_mask).float(),
            'frame_id': frame_id
        }

        self.cache[frame_id] = data
        return data

# -------------------------- 训练主循环
def train_model():
    OUTPUT_DIR = "./weights"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 配置项 ==========
    BATCH_SIZE = 1
    EPOCHS = 30
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # ========== 初始化数据集 ==========
    print("正在加载数据集...")
    full_kitti = KittiDataset(
        image_dir=IMAGE_DIR,
        point_dir=POINT_DIR,
        calib_dir=CALIB_DIR,
        label_dir=LABEL_DIR,
        index_filename=TRAIN_INDEX,
        is_training=True,
        difficulty=1,
        num_classes=4
    )

    # 简单的训练/验证划分
    num_total = len(full_kitti)
    num_train = int(0.8 * num_total)
    num_val = num_total - num_train
    train_indices = list(range(num_train))
    val_indices = list(range(num_train, num_total))

    # 子集包装类
    class SubsetKittiDataset:
        def __init__(self, full_dataset, indices):
            self.full_dataset = full_dataset
            self.indices = indices
            self.num_classes = full_dataset.num_classes

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]

        def get_calib(self, idx):
            return self.full_dataset.get_calib(self.indices[idx])

        def get_cam_points_in_image(self, idx, **kwargs):
            return self.full_dataset.get_cam_points_in_image(self.indices[idx], **kwargs)

        def get_label(self, idx):
            return self.full_dataset.get_label(self.indices[idx])

        def assign_car_label_to_points(self, *args, **kwargs):
            return self.full_dataset.assign_car_label_to_points(*args, **kwargs)

        def get_filename(self, idx):
            return self.full_dataset.get_filename(self.indices[idx])

    train_kitti_wrapped = SubsetKittiDataset(full_kitti, train_indices)
    val_kitti_wrapped = SubsetKittiDataset(full_kitti, val_indices)

    # PyTorch数据集封装
    train_dataset = CachedKittiGraphDataset(train_kitti_wrapped, is_training=True)
    val_dataset = CachedKittiGraphDataset(val_kitti_wrapped, is_training=False)
    NUM_WORKERS = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")

    # ========== 初始化模型、损失、优化器 ==========
    model = GraphTransformerDetector(
        in_dim=4, hidden_dim=128, num_layers=3, num_heads=4, num_classes=4
    ).to(DEVICE)
    criterion = ImprovedDetectionLoss(
        num_classes=4, cls_weight=0.05, box_weight=20.0, focal_gamma=2.0
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')

    # ========== 训练主循环 ==========
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_total_loss = 0.0
        train_cls_loss = 0.0
        train_box_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 训练")
        for batch in pbar:
            optimizer.zero_grad()

            # 数据移到设备
            xyz = batch['xyz'][0].to(DEVICE)
            attr = batch['attr'][0].to(DEVICE)
            edge_index = batch['edge_index'][0].to(DEVICE)
            cls_labels = batch['cls_labels'][0].to(DEVICE)
            box_labels = batch['box_labels'][0].to(DEVICE)
            valid_mask = batch['valid_mask'][0].to(DEVICE)

            # 前向传播
            cls_logits, box_pred = model(xyz, attr, edge_index)

            # 计算损失
            total_loss, cls_loss, box_loss = criterion(
                cls_logits, box_pred, cls_labels, box_labels, valid_mask
            )

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 统计
            train_total_loss += total_loss.item()
            train_cls_loss += cls_loss.item()
            train_box_loss += box_loss.item() if isinstance(box_loss, torch.Tensor) else box_loss

            pbar.set_postfix({
                "总损失": f"{total_loss.item():.4f}",
                "分类损失": f"{cls_loss.item():.4f}",
                "回归损失": f"{box_loss.item():.4f}" if isinstance(box_loss, torch.Tensor) else f"{box_loss:.4f}"
            })

        # 验证阶段
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 验证"):
                xyz = batch['xyz'][0].to(DEVICE)
                attr = batch['attr'][0].to(DEVICE)
                edge_index = batch['edge_index'][0].to(DEVICE)
                cls_labels = batch['cls_labels'][0].to(DEVICE)
                box_labels = batch['box_labels'][0].to(DEVICE)
                valid_mask = batch['valid_mask'][0].to(DEVICE)

                cls_logits, box_pred = model(xyz, attr, edge_index)
                total_loss, _, _ = criterion(cls_logits, box_pred, cls_labels, box_labels, valid_mask)
                val_total_loss += total_loss.item()

        # 打印结果
        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss = val_total_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} 结束:")
        print(f"训练平均损失: {train_avg_loss:.4f} | 验证平均损失: {val_avg_loss:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}\n")

        # 保存最优模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_graph_transformer_detector.pth")
            print("已保存最优模型")

        scheduler.step()

    print("训练完成!")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final_graph_transformer_detector.pth")

if __name__ == "__main__":
    train_model()
