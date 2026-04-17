import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms
from tqdm import tqdm
from collections import namedtuple

from src.data.dataset import SimpleKittiDataset
from src.models.graph_transformer import GraphTransformerDetector
from src.data.utils import build_adaptive_knn_graph, box3d_to_cam_points
from src.utils.visualization import visualize_bev

# 点云数据结构定义
Points = namedtuple('Points', ['xyz', 'attr'])

# ==================== 配置项 ====================
DATASET_ROOT = "/Users/你的用户名/Downloads/kitti"  # Mac用户改这里！
MODEL_PATH = "./weights/best_graph_transformer_detector.pth"
FRAME_IDX = 3
CONF_THRESH = 0.6
NMS_THRESH = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 推理函数 ====================
def inference_single_frame(model, dataset, idx, conf_thresh=0.5, nms_thresh=0.25):
    model.eval()
    with torch.no_grad():
        calib = dataset.get_calib(idx)
        points = dataset.get_cam_points_in_image(idx, 0.8, calib)
        gt_labels = dataset.get_label(idx)
        edge_index, _, _ = build_adaptive_knn_graph(points)

        xyz = torch.from_numpy(points.xyz).float().to(DEVICE)
        attr = torch.from_numpy(points.attr).float().to(DEVICE)
        edge_index = torch.from_numpy(edge_index).long().to(DEVICE)

        cls_logits, box_pred = model(xyz, attr, edge_index)
        cls_scores = torch.softmax(cls_logits, dim=-1)
        car_scores = cls_scores[:, 1] + cls_scores[:, 2]
        pos_mask = car_scores > conf_thresh

        if pos_mask.sum() == 0:
            return [], [l for l in gt_labels if l['name'] == 'Car'], points

        boxes = box_pred[pos_mask]
        scores = car_scores[pos_mask]

        # NMS去重(鸟瞰图视角)
        boxes_center_x = boxes[:, 0]
        boxes_center_z = boxes[:, 2]
        box_length = boxes[:, 3]
        box_width = boxes[:, 5]

        x1 = boxes_center_x - box_width / 2
        x2 = boxes_center_x + box_width / 2
        z1 = boxes_center_z - box_length / 2
        z2 = boxes_center_z + box_length / 2

        boxes_nms = torch.stack([x1, z1, x2, z2], dim=1)
        keep_idx = nms(boxes_nms, scores, nms_thresh)

        final_boxes = boxes[keep_idx].cpu().numpy()
        final_scores = scores[keep_idx].cpu().numpy()

        preds = []
        for box, score in zip(final_boxes, final_scores):
            x3d, y3d, z3d, l, h, w, yaw = box
            preds.append({
                'name':'Car',
                'x3d':x3d,
                'y3d':y3d,
                'z3d':z3d,
                'length':l,
                'height':h,
                'width':w,
                'yaw':yaw,
                'score':score
            })

        gts = [l for l in gt_labels if l['name'] == 'Car']
        return preds, gts, points

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("="*60)
    print("设置数据集路径...")
    print("="*60)

    # 自动找到存在的路径
    possible_image_dirs = [
        f"{DATASET_ROOT}/training/image_2",
        f"{DATASET_ROOT}/image_2",
        f"{DATASET_ROOT}/object/training/image_2",
        f"{DATASET_ROOT}/kitti/training/image_2"
    ]
    possible_point_dirs = [
        f"{DATASET_ROOT}/training/velodyne",
        f"{DATASET_ROOT}/velodyne",
        f"{DATASET_ROOT}/object/training/velodyne",
        f"{DATASET_ROOT}/kitti/training/velodyne"
    ]
    possible_calib_dirs = [
        f"{DATASET_ROOT}/training/calib",
        f"{DATASET_ROOT}/calib",
        f"{DATASET_ROOT}/object/training/calib",
        f"{DATASET_ROOT}/kitti/training/calib"
    ]
    possible_label_dirs = [
        f"{DATASET_ROOT}/training/label_2",
        f"{DATASET_ROOT}/label_2",
        f"{DATASET_ROOT}/object/training/label_2",
        f"{DATASET_ROOT}/kitti/training/label_2"
    ]

    IMAGE_DIR = None
    for d in possible_image_dirs:
        if os.path.exists(d):
            IMAGE_DIR = d
            break

    POINT_DIR = None
    for d in possible_point_dirs:
        if os.path.exists(d):
            POINT_DIR = d
            break

    CALIB_DIR = None
    for d in possible_calib_dirs:
        if os.path.exists(d):
            CALIB_DIR = d
            break

    LABEL_DIR = None
    for d in possible_label_dirs:
        if os.path.exists(d):
            LABEL_DIR = d
            break

    print(f"根目录: {DATASET_ROOT}")
    print(f"图像: {IMAGE_DIR} (存在: {os.path.exists(IMAGE_DIR)})")
    print(f"点云: {POINT_DIR} (存在: {os.path.exists(POINT_DIR)})")
    print(f"标定: {CALIB_DIR} (存在: {os.path.exists(CALIB_DIR)})")
    print(f"标签: {LABEL_DIR} (存在: {os.path.exists(LABEL_DIR)})")

    if not (IMAGE_DIR and POINT_DIR and CALIB_DIR and LABEL_DIR):
        print("\n❌ 错误:找不到数据文件!")
        print("正在列出根目录下的内容:")
        for item in os.listdir(DATASET_ROOT):
            print(f" - {item}")
        sys.exit(1)

    print("\n✅ 路径配置成功!")

    print("\n" + "="*60)
    print("开始加载模型与数据...")
    print("="*60)
    print(f"使用设备: {DEVICE}")

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️ 警告:模型文件不存在: {MODEL_PATH}")
        print("正在列出 ./weights/ 下的文件:")
        if os.path.exists("./weights/"):
            for f in os.listdir("./weights/"):
                print(f" - {f}")
        print("\n由于没有训练好的模型,我们将只加载数据集并查看点云。")

        # 只看数据
        dataset = SimpleKittiDataset(IMAGE_DIR, POINT_DIR, CALIB_DIR, LABEL_DIR)
        print(f"\n数据集加载成功,共 {len(dataset)} 帧")

        # 画一帧点云看看
        idx = 0
        calib = dataset.get_calib(idx)
        points = dataset.get_cam_points_in_image(idx, 0.8, calib)
        gts = [l for l in dataset.get_label(idx) if l['name'] == 'Car']
        print(f"\n正在可视化第 {idx} 帧点云与真实框...")
        visualize_bev([], gts, points)
    else:
        # 加载模型
        print("正在加载模型...")
        model = GraphTransformerDetector(
            in_dim=4, hidden_dim=128, num_layers=3, num_heads=4, num_classes=4
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("✅ 模型加载成功!")

        # 加载数据
        dataset = SimpleKittiDataset(IMAGE_DIR, POINT_DIR, CALIB_DIR, LABEL_DIR)
        print(f"✅ 数据集加载成功,共 {len(dataset)} 帧")

        # 推理
        print(f"\n正在推理第 {FRAME_IDX} 帧...")
        preds, gts, points = inference_single_frame(
            model, dataset, FRAME_IDX, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH
        )

        print(f"\n检测结果:")
        print(f" 预测框数量: {len(preds)}")
        print(f" 真实框数量: {len(gts)}")
        for i, det in enumerate(preds):
            print(f" 目标{i+1}: 置信度={det['score']:.4f}")

        # 可视化
        print("\n正在生成可视化结果...")
        visualize_bev(preds, gts, points)
