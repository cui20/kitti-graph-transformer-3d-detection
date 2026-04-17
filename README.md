# KITTI 3D Object Detection: End-to-End Graph Transformer Model
> Computer Vision Course Project | Implements a LiDAR 3D object detection model based on Graph Transformer for car detection on the KITTI dataset

## 📋 Project Overview
This project is the final assignment for the 3D Computer Vision course. It addresses the problem of LiDAR point cloud object detection in autonomous driving scenarios by designing and implementing an end-to-end Graph Transformer detection model. Key innovations include:
- Adaptive K-nearest neighbor graph construction to handle uneven point cloud density
- Graph Transformer layer enhanced with relative position encoding
- Hybrid loss function combining Focal Loss and Huber Loss
- Complete training, inference, and visualization pipeline for the KITTI dataset

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
