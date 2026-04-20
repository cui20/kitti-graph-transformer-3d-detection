# KITTI 3D Object Detection: End-to-End Graph Transformer Model
> Computer Vision Course Project | Implements a LiDAR 3D object detection model based on Graph Transformer for car detection on the KITTI dataset

## 📖 Project Overview
This is the final assignment for the 3D Computer Vision course. We address the problem of 3D object detection from LiDAR point clouds in autonomous driving scenarios by designing and implementing an end-to-end Graph Transformer detection model.

Unlike traditional voxel-based or point-based methods, our approach directly constructs graphs on raw point clouds and leverages Graph Transformer layers to capture both local geometric structures and global contextual information, achieving competitive performance on the KITTI benchmark.

### ✨ Key Innovations
- **Adaptive K-nearest neighbor graph construction**: Dynamically adjusts the number of neighbors based on local point density to effectively handle the uneven distribution of LiDAR point clouds
- **Graph Transformer layer enhanced with relative position encoding**: Improves spatial relationship modeling by incorporating 3D relative coordinates into the attention mechanism
- **Hybrid loss function**: Combines Focal Loss for classification and Huber Loss for bounding box regression to balance training stability and detection accuracy
- **Complete end-to-end pipeline**: Implements full training, validation, inference, and visualization workflows specifically tailored for the KITTI 3D object detection dataset

## 🛠️ Environment Setup
- **System**: macOS/Linux/Windows
- **Python**: 3.12
- **CUDA**: 12.4 (required for NVIDIA GPU, Mac can run inference on CPU)

### Dependency Installation
```bash
# Install PyTorch first (Mac version)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# Install PyTorch Geometric (Mac version)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Dataset Preparation
We use the KITTI 3D Object Detection Dataset.
Download Required Files
Dataset Directory Structure
Organize the downloaded files into the following directory structure:
```bash
data/
└── kitti/
    ├── training/
    │   ├── calib/          # Contains 7481 calibration files (.txt)
    │   ├── velodyne/       # Contains 7481 point cloud files (.bin)
    │   └── label_2/        # Contains 7481 label files (.txt)
    └── testing/
        ├── calib/          # Contains 7518 calibration files (.txt)
        └── velodyne/       # Contains 7518 point cloud files (.bin)
```

