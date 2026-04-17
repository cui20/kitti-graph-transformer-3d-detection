import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# -------------------------- 相对位置编码
class RelativePositionEncoding(nn.Module):
    def __init__(self, hidden_dim, pos_dim=3):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, xyz, edge_index):
        i, j = edge_index
        rel_pos = xyz[j] - xyz[i]
        edge_attr = self.pos_mlp(rel_pos)
        return edge_attr

# -------------------------- 图Transformer层
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.transformer_conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            beta=True,
            concat=True,
            dropout=dropout,
            edge_dim=out_dim
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 自注意力 + 残差
        x_res = x
        x = self.transformer_conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x + x_res)

        # FFN + 残差
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x + x_res)
        return x

# -------------------------- 完整检测模型
class GraphTransformerDetector(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, num_layers=3, num_heads=4, num_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1. 点特征初始编码
        self.input_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 2. 相对位置编码(作为边属性)
        self.pos_encoding = RelativePositionEncoding(hidden_dim)

        # 3. 堆叠图Transformer层
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(
                GraphTransformerLayer(hidden_dim, hidden_dim, num_heads)
            )

        # 4. 检测头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7)
        )

    def forward(self, xyz, attr, edge_index):
        # 1. 初始特征编码
        x = torch.cat([xyz, attr], dim=-1)
        x = self.input_encoder(x)

        # 2. 计算相对位置编码(作为边属性)
        edge_attr = self.pos_encoding(xyz, edge_index)

        # 3. 图Transformer特征提取
        for layer in self.transformer_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)

        # 4. 检测头预测
        cls_logits = self.cls_head(x)
        box_pred = self.box_head(x)
        return cls_logits, box_pred

# -------------------------- 损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ImprovedDetectionLoss(nn.Module):
    def __init__(self, num_classes=4, cls_weight=0.05, box_weight=20.0, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.cls_loss = FocalLoss(gamma=focal_gamma, ignore_index=3)
        self.box_loss = nn.HuberLoss(reduction='none')

    def forward(self, cls_logits, box_pred, cls_labels, box_labels, valid_mask):
        # 分类损失 (Focal Loss)
        cls_loss = self.cls_loss(cls_logits, cls_labels.squeeze(-1))

        # 回归损失:仅计算正样本
        pos_mask = (cls_labels.squeeze(-1) == 1) | (cls_labels.squeeze(-1) == 2)
        pos_mask = pos_mask & valid_mask.squeeze(-1).bool()

        if pos_mask.sum() > 0:
            box_loss = self.box_loss(box_pred[pos_mask], box_labels[pos_mask]).mean()
        else:
            box_loss = 0.0

        # 总损失
        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss
        return total_loss, cls_loss, box_loss
