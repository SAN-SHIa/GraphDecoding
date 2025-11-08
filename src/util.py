import os
import random
import logging
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import shortest_path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, '{}_{}.log'.format(datetime.datetime.now().strftime('%m%d%H%M'), name))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

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

def dG(A, B):
    S = A.T @ B
    U, Sigma, V = torch.svd(S)
    R = U @ V.T
    AR = A @ R
    return ((AR - B) ** 2).sum(1).mean()
