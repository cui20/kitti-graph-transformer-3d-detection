# KITTI 3D Object Detection: End-to-End Graph Transformer Model
> Computer Vision Course Project | Implements a LiDAR 3D object detection model based on Graph Transformer for car detection on the KITTI dataset

📖 Project Overview
This is the final assignment for the 3D Computer Vision course. We address the problem of 3D object detection from LiDAR point clouds in autonomous driving scenarios by designing and implementing an end-to-end Graph Transformer detection model.
Unlike traditional voxel-based or point-based methods, our approach directly constructs graphs on raw point clouds and leverages Graph Transformer layers to capture both local geometric structures and global contextual information, achieving competitive performance on the KITTI benchmark.
✨ Key Innovations
Adaptive K-nearest neighbor graph construction: Dynamically adjusts the number of neighbors based on local point density to effectively handle the uneven distribution of LiDAR point clouds
Graph Transformer layer enhanced with relative position encoding: Improves spatial relationship modeling by incorporating 3D relative coordinates into the attention mechanism
Hybrid loss function: Combines Focal Loss for classification and Huber Loss for bounding box regression to balance training stability and detection accuracy
Complete end-to-end pipeline: Implements full training, validation, inference, and visualization workflows specifically tailored for the KITTI 3D object detection dataset

## 🛠️ Environment Setup
- System: macOS/Linux/Windows
- Python: 3.12
- CUDA: 12.4 (required for NVIDIA GPU, Mac can run inference on CPU)
- Dependency Installation:
```bash
# Install PyTorch first (Mac version)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# Install PyTorch Geometric (Mac version)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install other dependencies
pip install -r requirements.txt

📊 Dataset Preparation
We use the KITTI 3D Object Detection Dataset.
Download Required Files
Go to the KITTI 3D Object Detection Benchmark page
Download the following files:
data_object_calib.zip (Calibration files, 16MB)
data_object_velodyne.zip (LiDAR point clouds, 14GB)
data_object_label_2.zip (Training labels, 5MB)
Dataset Directory Structure
Organize the downloaded files into the following directory structure:
data/
└── kitti/
    ├── training/
    │   ├── calib/          # Contains 7481 calibration files (.txt)
    │   ├── velodyne/       # Contains 7481 point cloud files (.bin)
    │   └── label_2/        # Contains 7481 label files (.txt)
    └── testing/
        ├── calib/          # Contains 7518 calibration files (.txt)
        └── velodyne/       # Contains 7518 point cloud files (.bin)
Data Preprocessing
No additional preprocessing is required. The dataset loader will automatically process the raw point clouds and labels during training.
graph-transformer-3d-detection/
├── data/                   # Dataset directory (not included in repository)
├── checkpoints/            # Saved model checkpoints
├── results/                # Inference and visualization results
├── docs/                   # Documentation and result images
├── models/                 # Model implementation
│   ├── graph_transformer.py  # Core Graph Transformer layer with relative position encoding
│   ├── backbone.py           # Point cloud feature extraction backbone
│   ├── detector.py           # Detection head for classification and bounding box regression
│   └── adaptive_knn.py       # Adaptive K-nearest neighbor graph construction module
├── datasets/               # Dataset loading and processing
│   └── kitti_dataset.py      # KITTI dataset class and data augmentation
├── utils/                  # Utility functions
│   ├── loss.py               # Hybrid loss function implementation
│   ├── metrics.py            # 3D detection evaluation metrics (mAP calculation)
│   ├── visualization.py      # Point cloud and detection result visualization tools
│   └── config.py             # Configuration parameters
├── train.py                # Training script
├── val.py                  # Validation and evaluation script
├── infer.py                # Single sample inference script
├── visualize.py            # Detection result visualization script
├── requirements.txt        # Complete list of dependencies
└── README.md               # This file
