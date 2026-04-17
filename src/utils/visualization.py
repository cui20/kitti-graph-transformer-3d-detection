import matplotlib.pyplot as plt
import numpy as np
from src.data.utils import box3d_to_cam_points

def visualize_bev(preds, gts, points, xlim=(-40,40), zlim=(0,70), figsize=(12,8)):
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 画点云
    bev_xyz = points.xyz
    mask = (bev_xyz[:,0]>xlim[0]) & (bev_xyz[:,0]<xlim[1]) & \
           (bev_xyz[:,2]>zlim[0]) & (bev_xyz[:,2]<zlim[1])
    ax.scatter(bev_xyz[mask,0], bev_xyz[mask,2], s=0.5, c='gray', alpha=0.5, label='Point Cloud')

    # 画真实框
    for det in gts:
        box_corners = box3d_to_cam_points(det).xyz[:, [0,2]]
        for i in range(4):
            start, end = box_corners[i], box_corners[(i+1)%4]
            ax.plot([start[0], end[0]], [start[1], end[1]], c='red', linewidth=2,
                    label='Ground Truth' if i == 0 else "")

    # 画预测框
    for det in preds:
        box_corners = box3d_to_cam_points(det).xyz[:, [0,2]]
        for i in range(4):
            start, end = box_corners[i], box_corners[(i+1)%4]
            ax.plot([start[0], end[0]], [start[1], end[1]], c='lime', linewidth=2, linestyle='--',
                    label=f'Pred (conf={det["score"]:.2f})' if i == 0 else "")

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('BEV Detection Results', fontsize=14)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_raw_data(dataset, idx):
    calib = dataset.get_calib(idx)
    points = dataset.get_cam_points_in_image(idx, 0.8, calib)
    all_labels = dataset.get_label(idx)

    print(f"第 {idx} 帧的所有标注:")
    for i, label in enumerate(all_labels):
        print(f" 标注 {i+1}: 类别={label['name']}, 截断={label['truncation']:.2f}, 遮挡={label['occlusion']}")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 画点云
    bev_xyz = points.xyz
    mask = (bev_xyz[:,0]>-40) & (bev_xyz[:,0]<40) & (bev_xyz[:,2]>0) & (bev_xyz[:,2]<70)
    ax.scatter(bev_xyz[mask,0], bev_xyz[mask,2], s=0.5, c='gray', alpha=0.5, label='Point Cloud')

    # 画所有类别的真实框
    color_map = {'Car':'red', 'Van':'orange', 'Truck':'purple', 'Pedestrian':'blue', 'Cyclist':'green'}
    for det in all_labels:
        color = color_map.get(det['name'], 'black')
        box_corners = box3d_to_cam_points(det).xyz[:, [0,2]]
        for i in range(4):
            start, end = box_corners[i], box_corners[(i+1)%4]
            ax.plot([start[0], end[0]], [start[1], end[1]], c=color, linewidth=2,
                    label=f"{det['name']}" if i == 0 else "")

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title(f'Frame {idx}: Raw Point Cloud & All Ground Truth', fontsize=14)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
