import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import shortest_path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GATConv, SAGEConv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, input):
        return self.main(input)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=128):
        super(MLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.LeakyReLU(),
            nn.Linear(mid_channels, out_channels),
        )

    def forward(self, input):
        x = self.main(input)
        return x


class GIN(torch.nn.Module):
    def __init__(self, m):
        super(GIN, self).__init__()

        def init_weights(layer):
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(0, 1e-3)

        self.mlp1 = MLP(m + 2, 128)
        self.conv1 = GINConv(self.mlp1)
        self.mlp1.main.apply(init_weights)
        self.mlp2 = MLP(128, 128)
        self.conv2 = GINConv(self.mlp2)
        self.mlp2.main.apply(init_weights)
        self.mlp3 = MLP(128, 2)
        self.conv3 = GINConv(self.mlp3)
        self.mlp3.main.apply(init_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, m):
        super(GAT, self).__init__()
        self.conv1 = GATConv(m + 2, 16, heads=8)
        if hasattr(self.conv1, 'lin_src') and self.conv1.lin_src is not None:
            self.conv1.lin_src.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv1, 'lin_dst') and self.conv1.lin_dst is not None:
            self.conv1.lin_dst.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv1, 'lin') and self.conv1.lin is not None:
            self.conv1.lin.weight.data.normal_(0, 1e-3)
            
        self.conv2 = GATConv(128, 16, heads=8)
        if hasattr(self.conv2, 'lin_src') and self.conv2.lin_src is not None:
            self.conv2.lin_src.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv2, 'lin_dst') and self.conv2.lin_dst is not None:
            self.conv2.lin_dst.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv2, 'lin') and self.conv2.lin is not None:
            self.conv2.lin.weight.data.normal_(0, 1e-3)
            
        self.conv3 = GATConv(128, 2, heads=1)
        if hasattr(self.conv3, 'lin_src') and self.conv3.lin_src is not None:
            self.conv3.lin_src.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv3, 'lin_dst') and self.conv3.lin_dst is not None:
            self.conv3.lin_dst.weight.data.normal_(0, 1e-3)
        if hasattr(self.conv3, 'lin') and self.conv3.lin is not None:
            self.conv3.lin.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x


# yes 方案六：自适应融合 (AdaFusion / AdaIN style)
# 利用密度生成缩放和平移因子，对特征进行仿射变换
class GraphSAGE_AdaFusion(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_AdaFusion, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        self.scale1 = nn.Linear(3, 128) # Density dim 1 -> 3
        self.shift1 = nn.Linear(3, 128)

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        self.scale2 = nn.Linear(3, 128)
        self.shift2 = nn.Linear(3, 128)

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3] # Slice last 3 dims
        density = x[:, -3:]
        
        x = self.conv1(features, edge_index)
        # AdaIN: x = x * (1 + scale) + shift
        s1 = self.scale1(density)
        b1 = self.shift1(density)
        x = x * (1 + s1) + b1
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        s2 = self.scale2(density)
        b2 = self.shift2(density)
        x = x * (1 + s2) + b2
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# yes 方案七：自适应门控融合 (AdaGating)
# 结合 AdaIN 的仿射变换和 Gating 的非线性截断，双重调制
class GraphSAGE_AdaGating(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_AdaGating, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        self.scale1 = nn.Linear(3, 128)
        self.shift1 = nn.Linear(3, 128)
        self.gate1 = nn.Sequential(nn.Linear(3, 128), nn.Sigmoid())

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        self.scale2 = nn.Linear(3, 128)
        self.shift2 = nn.Linear(3, 128)
        self.gate2 = nn.Sequential(nn.Linear(3, 128), nn.Sigmoid())

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        x = self.conv1(features, edge_index)
        # AdaIN
        s1 = self.scale1(density)
        b1 = self.shift1(density)
        x = x * (1 + s1) + b1
        # Gating
        g1 = self.gate1(density)
        x = x * g1
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        s2 = self.scale2(density)
        b2 = self.shift2(density)
        x = x * (1 + s2) + b2
        g2 = self.gate2(density)
        x = x * g2
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# yes 方案八：多尺度自适应融合 (Multi-Scale AdaFusion)
# 扩展密度特征，使用 [d, d^2, sqrt(d)] 作为条件输入，增强非线性表达
class GraphSAGE_MultiScaleAda(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_MultiScaleAda, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        # 输入维度变为 3 (Stationary, InDegree, Clustering)
        self.scale1 = nn.Linear(3, 128)
        self.shift1 = nn.Linear(3, 128)

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        self.scale2 = nn.Linear(3, 128)
        self.shift2 = nn.Linear(3, 128)

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        # 直接使用3维密度特征
        d_multi = density
        
        x = self.conv1(features, edge_index)
        s1 = self.scale1(d_multi)
        b1 = self.shift1(d_multi)
        x = x * (1 + s1) + b1
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        s2 = self.scale2(d_multi)
        b2 = self.shift2(d_multi)
        x = x * (1 + s2) + b2
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# no2 方案九：注意力自适应融合 (Attention AdaFusion)
# 利用密度生成通道注意力权重，对特征进行加权，再进行 AdaIN
class GraphSAGE_AttentionAda(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_AttentionAda, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        self.attn1 = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Sigmoid())
        self.scale1 = nn.Linear(3, 128)
        self.shift1 = nn.Linear(3, 128)

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        self.attn2 = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Sigmoid())
        self.scale2 = nn.Linear(3, 128)
        self.shift2 = nn.Linear(3, 128)

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        x = self.conv1(features, edge_index)
        
        # Channel Attention based on density
        attn1 = self.attn1(density)
        x = x * attn1
        
        # AdaIN
        s1 = self.scale1(density)
        b1 = self.shift1(density)
        x = x * (1 + s1) + b1
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        # Channel Attention
        attn2 = self.attn2(density)
        x = x * attn2
        
        # AdaIN
        s2 = self.scale2(density)
        b2 = self.shift2(density)
        x = x * (1 + s2) + b2
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# 方案十：GIN + Attention Ada (Optimized)
# 增加 BatchNorm 防止梯度爆炸，初始化 AdaIN 参数为 0 以稳定训练起始点
class GIN_AttentionAda(torch.nn.Module):
    def __init__(self, m):
        super(GIN_AttentionAda, self).__init__()

        def init_weights(layer):
            if type(layer) == nn.Linear:
                layer.weight.data.normal_(0, 1e-3)

        # Layer 1
        self.mlp1 = MLP(m + 2, 128)
        self.mlp1.main.apply(init_weights)
        self.conv1 = GINConv(self.mlp1)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.attn1 = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Sigmoid())
        self.scale1 = nn.Linear(3, 128)
        self.shift1 = nn.Linear(3, 128)
        
        # Zero init for stability
        self.scale1.weight.data.fill_(0)
        self.scale1.bias.data.fill_(0)
        self.shift1.weight.data.fill_(0)
        self.shift1.bias.data.fill_(0)

        # Layer 2
        self.mlp2 = MLP(128, 128)
        self.mlp2.main.apply(init_weights)
        self.conv2 = GINConv(self.mlp2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.attn2 = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Sigmoid())
        self.scale2 = nn.Linear(3, 128)
        self.shift2 = nn.Linear(3, 128)

        # Zero init for stability
        self.scale2.weight.data.fill_(0)
        self.scale2.bias.data.fill_(0)
        self.shift2.weight.data.fill_(0)
        self.shift2.bias.data.fill_(0)

        # Layer 3
        self.mlp3 = MLP(128, 2)
        self.mlp3.main.apply(init_weights)
        self.conv3 = GINConv(self.mlp3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        # Layer 1
        x = self.conv1(features, edge_index)
        x = self.bn1(x)
        
        # Attention & AdaIN
        attn1 = self.attn1(density)
        s1 = self.scale1(density)
        b1 = self.shift1(density)
        x = x * attn1 * (1 + s1) + b1
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        # Attention & AdaIN
        attn2 = self.attn2(density)
        s2 = self.scale2(density)
        b2 = self.shift2(density)
        x = x * attn2 * (1 + s2) + b2
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        return x



# no1 方案十九：简单缩放 (SimpleScale)
# 仅使用密度进行缩放 (Scaling)，不进行平移 (Shift)，简化版 AdaIN
class GraphSAGE_SimpleScale(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_SimpleScale, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        self.scale1 = nn.Linear(3, 128)

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        self.scale2 = nn.Linear(3, 128)

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        x = self.conv1(features, edge_index)
        s1 = self.scale1(density)
        x = x * (1 + s1)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        s2 = self.scale2(density)
        x = x * (1 + s2)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# 方案二十六：修正缩放 (CorrectiveScale) - 1+1>2 方案 (最终优化版)
# 核心思想：以 SimpleScale (线性缩放) 为主干，增加一个受门控控制的非线性修正项
# 结构：Scale = S_base(Linear) + S_fine(MLP) * Gate(Sigmoid)
# 优势：
# 1. 继承 SimpleScale 的稳定性：Gate 初始化或学习为 0 时，退化为 SimpleScale
# 2. 吸收 AttentionAda 的非线性：S_fine 提供非线性拟合能力
# 3. 引入选择机制：Gate 允许模型根据密度动态决定是否启用非线性修正
class GraphSAGE_CorrectiveScale(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_CorrectiveScale, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv1.lin_l.weight.data.normal_(0, 1e-3)
        self.conv1.lin_r.weight.data.normal_(0, 1e-3)
        
        # Base Linear Scale (Robust backbone)
        self.scale1_base = nn.Linear(3, 128)
        # Fine-tuning Non-linear Scale (Correction)
        self.scale1_fine = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 128), nn.Tanh()
        )
        # Gating mechanism (Selection)
        self.gate1 = nn.Sequential(nn.Linear(3, 128), nn.Sigmoid())

        self.conv2 = SAGEConv(128, 128)
        self.conv2.lin_l.weight.data.normal_(0, 1e-3)
        self.conv2.lin_r.weight.data.normal_(0, 1e-3)
        
        self.scale2_base = nn.Linear(3, 128)
        self.scale2_fine = nn.Sequential(
            nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 128), nn.Tanh()
        )
        self.gate2 = nn.Sequential(nn.Linear(3, 128), nn.Sigmoid())

        self.conv3 = SAGEConv(128, 2)
        self.conv3.lin_l.weight.data.normal_(0, 1e-3)
        self.conv3.lin_r.weight.data.normal_(0, 1e-3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        x = self.conv1(features, edge_index)
        
        # Calculate Scale components
        s_base = self.scale1_base(density)
        s_fine = self.scale1_fine(density)
        g = self.gate1(density)
        
        # Apply Scale: x * (1 + Base + Fine * Gate)
        x = x * (1 + s_base + s_fine * g)
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        s_base2 = self.scale2_base(density)
        s_fine2 = self.scale2_fine(density)
        g2 = self.gate2(density)
        
        x = x * (1 + s_base2 + s_fine2 * g2)
        
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# 方案 A：输入投影融合 (Input Projection)
# 思路：不在层间做复杂的数学调制，而是将密度映射到高维空间，直接作为初始特征的一部分。
# 这种方法最简单，让 GNN 自己去学习密度和图结构的关系。
class GraphSAGE_InputProject(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_InputProject, self).__init__()
        # 将密度(3维)映射到 64 维
        self.density_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 64)
        )
        
        # 输入维度 = 原特征维度(m+2) + 密度编码维度(64)
        # 修正：features 是 m+2 维 (X[2] + ind[m])
        self.conv1 = SAGEConv((m + 2) + 64, 128)
        self.conv2 = SAGEConv(128, 128)
        self.conv3 = SAGEConv(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3] # [N, m+2]
        density = x[:, -3:]  # [N, 3]
        
        d_emb = self.density_encoder(density)
        # 早期强融合
        x_in = torch.cat([features, d_emb], dim=1)
        
        x = self.conv1(x_in, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


# 方案 B：跳跃密度连接 (Skip Density)
# 思路：密度信息不参与卷积聚合（防止被平滑），而是作为一种“加性偏置”，
# 在每一层卷积后以残差形式加入。
class GraphSAGE_SkipDensity(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_SkipDensity, self).__init__()
        self.conv1 = SAGEConv(m + 2, 128)
        self.dense_proj1 = nn.Linear(3, 128)
        
        self.conv2 = SAGEConv(128, 128)
        self.dense_proj2 = nn.Linear(3, 128)
        
        self.conv3 = SAGEConv(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        # Layer 1
        h = self.conv1(features, edge_index)
        d = self.dense_proj1(density)
        h = h + d # Additive fusion
        h = F.leaky_relu(h)
        h = F.dropout(h, training=self.training)
        
        # Layer 2
        h = self.conv2(h, edge_index)
        d = self.dense_proj2(density)
        h = h + d
        h = F.leaky_relu(h)
        h = F.dropout(h, training=self.training)
        
        # Layer 3
        h = self.conv3(h, edge_index)
        return h


# 方案 C：双流分离处理 (Separated Stream)
# 思路：一路处理图结构特征，一路仅处理密度（MLP）。最后才融合。
# 避免图卷积过度平滑密度带来的位置先验信息。
class GraphSAGE_Separated(torch.nn.Module):
    def __init__(self, m):
        super(GraphSAGE_Separated, self).__init__()
        # Graph Branch
        # 修正：features 是 m+2 维
        self.conv1 = SAGEConv(m + 2, 128)
        self.conv2 = SAGEConv(128, 128)
        
        # Density Branch (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU()
        )
        
        # Fusion
        self.final = nn.Linear(128 + 128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        features = x[:, :-3]
        density = x[:, -3:]
        
        # Graph Path
        h_g = self.conv1(features, edge_index)
        h_g = F.leaky_relu(h_g)
        h_g = F.dropout(h_g, training=self.training)
        h_g = self.conv2(h_g, edge_index)
        h_g = F.leaky_relu(h_g)
        
        # Density Path
        h_d = self.mlp(density)
        
        # Concatenate
        h = torch.cat([h_g, h_d], dim=1)
        return self.final(h)


# 方案 D：GAT 密度增强 (GAT Density Enhanced)
# 思路：使用 GAT，并将密度作为一种强特征直接拼接到每一层的输入中。
# GAT 的注意力机制可能会根据密度差异自动调整权重。
class GAT_DensityEnhanced(torch.nn.Module):
    def __init__(self, m):
        super(GAT_DensityEnhanced, self).__init__()
        # Input: features(m+2) + density(3) = m+5
        # 修正：输入总维度是 m+5
        self.conv1 = GATConv(m + 5, 16, heads=8) # out: 16*8 = 128
        
        # Input: 128 + density(3) = 131
        self.conv2 = GATConv(131, 16, heads=8) # out: 128
        
        # Input: 128 + density(3) = 131
        self.conv3 = GATConv(131, 2, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x already contains density at the end
        density = x[:, -3:]
        
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, training=self.training)
        
        # Re-append density for next layer
        h = torch.cat([h, density], dim=1)
        
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        h = F.dropout(h, training=self.training)
        
        # Re-append density
        h = torch.cat([h, density], dim=1)
        
        h = self.conv3(h, edge_index)
        return h


def moon(n):
    m = int(n / 2)
    t = np.pi * np.random.rand(2 * m, 1)
    x = 6 * np.cos(t)
    y = 6 * np.sin(t)
    z = np.hstack([x, y])
    a = np.random.randn(m, 2) + z[:m]
    b = np.random.randn(m, 2) + np.array([6, 0]) + z[m:] * np.array([1, -1])
    x = np.concatenate([a, b]) * 0.15
    n = len(x)
    return x, n


def stationary(A):
    eig = eigs(A.T)
    ind = eig[0].real.argsort()[-1]
    est = eig[1][:, ind].real
    pr = est / est.sum() * A.shape[0]
    return pr


def reconstruct(K, pr, n, m, fr, to):
    selected = np.random.choice(np.arange(n), m, replace=False)
    unselected = np.array(list(set(np.arange(n)) - set(selected)))
    s = (K / (pr * n * n)) ** 0.25
    W = csr_matrix((s.repeat(K), (fr, to)))
    spd = shortest_path(W, indices=selected)
    pos_inf = (spd == np.inf)
    spd[pos_inf] = 0
    spd[pos_inf] = spd.max()
    selected_spd = spd[:, selected]
    sspd = (selected_spd + selected_spd.T) / 2
    sspd = sspd ** 2
    H = np.eye(m) - np.ones(m) / n
    Ker = - H @ sspd @ H / 2
    w, v = np.linalg.eigh(Ker)
    rec_unnormalized = v[:, -2:] @ np.diag(w[-2:])
    rec_orig = np.zeros((n, 2))
    rec_orig[selected] = rec_unnormalized
    rec_orig[unselected] = rec_unnormalized[spd[:, unselected].argmin(0)]
    return rec_orig


def reconstruct_full(dim, deg, pr, n, m, fr, to):
    selected = np.random.choice(np.arange(n), m, replace=False)
    unselected = np.array(list(set(np.arange(n)) - set(selected)))
    s = (deg / (pr * n * n)) ** 0.25
    W = csr_matrix(([s[x] for x in fr], (fr, to)))
    spd = shortest_path(W, indices=selected)
    pos_inf = (spd == np.inf)
    spd[pos_inf] = 0
    spd[pos_inf] = spd.max()
    selected_spd = spd[:, selected]
    sspd = (selected_spd + selected_spd.T) / 2
    sspd = sspd ** 2
    H = np.eye(m) - np.ones(m) / n
    Ker = - H @ sspd @ H / 2
    w, v = np.linalg.eigh(Ker)
    rec_unnormalized = v[:, -dim:] @ np.diag(w[-dim:])
    rec_orig = np.zeros((n, dim))
    rec_orig[selected] = rec_unnormalized
    rec_orig[unselected] = rec_unnormalized[spd[:, unselected].argmin(0)]
    return rec_orig


def dG(A, B):
    S = A.T @ B
    U, Sigma, V = torch.svd(S)
    R = U @ V.T
    AR = A @ R
    return ((AR - B) ** 2).sum(1).mean()
