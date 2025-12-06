import os
import random
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import orthogonal_procrustes

import torch
import torch.optim as optim
from torch_geometric.data import Data

from sklearn.manifold import TSNE
from util import Net, GIN, GAT, GraphSAGE, GraphSAGE_SimpleScale, GraphSAGE_CorrectiveScale, GraphSAGE_InputProject, GraphSAGE_SkipDensity, GraphSAGE_Separated, GAT_DensityEnhanced, moon, stationary, reconstruct,GraphSAGE_AttentionAda, dG
import logging
import tqdm
import datetime

os.makedirs('logs', exist_ok=True)
LOG_FILE = 'logs/{}_semi_moon.log'.format(datetime.datetime.now().strftime('%m%d%H%M'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("semi_moon")


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(0)

n = 5000
m = 500
x, n = moon(n)
n_train = int(n * 0.7)
train_ind = torch.randperm(n)[:n_train]
test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
K = int(np.sqrt(n) * np.log2(n) / 10)
D = pairwise_distances(x)
fr = np.arange(n).repeat(K).reshape(-1)
to = np.argsort(D, axis=1)[:, 1:K + 1].reshape(-1)
A = csr_matrix((np.ones(n * K) / K, (fr, to)))

edge_index = np.vstack([fr, to])
edge_index = torch.tensor(edge_index, dtype=torch.long)
X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)
eye_n = torch.eye(n)
logger.info(f"初始化完成: n={n}, m={m}, K={K}, n_train={n_train}")

# Calculate density for fusion methods
# 1. Stationary Distribution (Global)
pr = stationary(A)
pr = np.maximum(pr, 1e-9)

# 2. In-Degree (Local Connectivity)
# Construct NetworkX graph for calculation
G_nx = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
in_deg = np.array([G_nx.in_degree(i) for i in range(n)])
in_deg = in_deg / (np.mean(in_deg) + 1e-9) # Normalize to mean ~1

# 3. Clustering Coefficient (Local Cohesiveness)
clust = nx.clustering(G_nx)
clust = np.array([clust[i] for i in range(n)])
clust = clust / (np.mean(clust) + 1e-9) # Normalize to mean ~1

# Combine into 3D density feature: [Stationary, InDegree, Clustering]
density_np = np.vstack([pr, in_deg, clust]).T
density = torch.FloatTensor(density_np) # Shape: [N, 3]
x_tensor = torch.FloatTensor(x)



def run_experiment(model_name, model_cls, epochs=100, lr=0.001):
    seed_everything(0)
    net = model_cls(m)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    final_rec = None
    for _ in tqdm.trange(epochs, desc=model_name):
        idx = torch.randperm(n)[:m]
        ind = eye_n[:, idx]
        X_extended = torch.hstack([X, ind])
        X_with_density = torch.cat([X_extended, density], dim=1)
        data = Data(x=X_with_density, edge_index=edge_index)
        rec = net(data)
        loss = dG(x_tensor[train_ind], rec[train_ind])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_rec = rec
    rec_np = final_rec.detach().cpu().numpy()
    R, _ = orthogonal_procrustes(x, rec_np)
    aligned = rec_np @ R.T
    score = float(dG(x_tensor, torch.FloatTensor(aligned)))
    logger.info(f"✅ {model_name} training completed: dG={score:.4f}")
    return score


experiments = [
    # New Models replacing the complex ones
    # ("GraphSAGE_InputProject", GraphSAGE_InputProject, 100),
    # ("GraphSAGE_SkipDensity", GraphSAGE_SkipDensity, 100),
    # ("GraphSAGE_Separated", GraphSAGE_Separated, 100),
    # ("GAT_DensityEnhanced", GAT_DensityEnhanced, 100),
    # previous models
    ("GraphSAGE_SimpleScale", GraphSAGE_SimpleScale, 100),
    ("GraphSAGE_AttentionAda", GraphSAGE_AttentionAda, 100),
    # ("GraphSAGE_AttentionAdaPlus", GraphSAGE_AttentionAdaPlus, 120),
    # ("GraphSAGE_CorrectiveScale", GraphSAGE_CorrectiveScale, 120),

]
results = {}
for name, cls, epochs in experiments:
    results[name] = run_experiment(name, cls, epochs=epochs)
logger.info("📊 模型 dG 排行：")
for name, score in sorted(results.items(), key=lambda kv: kv[1]):
    logger.info(f"{name}: dG={score:.4f}")
