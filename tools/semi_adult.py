import os
import csv
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

from utils.model import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything
from utils.logging import setup_logger
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


np.random.seed(0)
torch.manual_seed(0)

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

logger.info(f"初始化完成: n={n}, m={m}, K={K}, n_train={n_train}")

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for i in tqdm.tqdm(range(100)):
    # Note 1: In the original formulation, $g$, i.e., the neural network for the scale function, should be used in reconstruct(K, pr, n, m, fr, to), namely, in the definition of $s$. We factorize $s$ and multiply g after we reconstruct the features. This is mathematically equivalent. We do this to avoid memory overflow due to long backpropagation.
    # Note 2: We roughly standardize n for stability by (n - 3000) / 3000. This does not affect the representational power of GNNs by merging them into the network parameters.
    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    rec_orig = reconstruct(K, pr, n, m, fr, to)
    rec_orig = torch.FloatTensor(rec_orig)
    g = net(torch.FloatTensor([(n - 3000) / 3000]))
    rec = rec_orig * (g ** 0.5)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])

    # print(n, float(g), float(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_proposed = rec.detach().numpy() @ R.T
loss_proposed = float(dG(torch.FloatTensor(x), torch.FloatTensor(rec_proposed)))

logger.info(f"✅ Proposed method training completed: dG={loss_proposed:.4f}")

net = GIN(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in tqdm.tqdm(range(100)):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    # print(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_GIN = rec.detach().numpy() @ R.T
loss_GIN = float(dG(torch.FloatTensor(x), torch.FloatTensor(rec_GIN)))

logger.info(f"✅ GIN method training completed: dG={loss_GIN:.4f}")

net = GAT(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in tqdm.tqdm(range(100)):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    # print(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# def prepare_data(n):
#     """Prepare moon dataset and distance matrix."""
#     x, n = moon(n)
#     n_train = int(n * 0.7)
#     train_ind = torch.randperm(n)[:n_train]
#     D = pairwise_distances(x)
#     return x, train_ind, D
def prepare_data(n=None):
    """Prepare adult dataset and distance matrix."""
    x = []
    with open('src/adult.data') as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) == 15 and int(r[0]) < 90 and 1000 < int(r[10]) and int(r[10]) < 99999:
                x.append([int(r[0]), np.log10(int(r[10]))])

    x = np.array(x)
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    x = (x - mu) / std
    n = len(x)
    
    n_train = int(n * 0.7)
    train_ind = torch.randperm(n)[:n_train]
    D = pairwise_distances(x)
    return x, train_ind, D, n

logger.info(f"✅ GAT training completed: dG={loss_GAT:.4f}")

ind = torch.eye(n)[:, torch.randperm(n)[:m]]
X_extended = torch.hstack([X, ind])
X_embedded = TSNE(n_components=2, random_state=0, init='pca').fit_transform(X_extended.numpy())
loss_tSNE = float(dG(torch.FloatTensor(x), torch.FloatTensor(X_embedded)))
logger.info(f"✅ tSNE embedding completed: dG={loss_tSNE:.4f}")

logger.info("🎨 visualization...")
c = x[:, 0].argsort().argsort()
fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(2, 3, 1)
ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#eeeeee')
txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

visible = plt.imread('./imgs/visible.png')
visible_ax = fig.add_axes([0.24, 0.77, 0.1, 0.1], anchor='NE', zorder=1)
visible_ax.imshow(visible)
visible_ax.axis('off')

def prepare_eball_features(D, n):
    """Build E-ball graph and compute its density features."""
    # 对于分散数据，使用更大的百分位数和缩放因子
    base_epsilon = np.percentile(D[D > 0], 5.0)  # 从2.0提高到5.0
    scaling_factor = 3.5  # 从2.7提高到3.5
    epsilon = base_epsilon * scaling_factor
    logger.info(f"Using scaling factor {scaling_factor}: epsilon={epsilon:.6f}")
    
    A_eball, edge_index_eball = build_eball_graph(D, epsilon)
    edge_index_eball = torch.tensor(edge_index_eball, dtype=torch.long)
    
    avg_degree = calculate_average_degree(A_eball)
    logger.info(f"🌟 E-ball Graph: Nodes={n}, Edges={A_eball.nnz}, Avg Degree={avg_degree:.4f}")
    
    pr_eball = stationary(A_eball)
    pr_eball = np.maximum(pr_eball, 1e-9)

ax = fig.add_subplot(2, 3, 3)
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'tSNE(X) $d_G = {:.2f}$'.format(loss_tSNE), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(2, 3, 4)
ax.scatter(rec_proposed[:, 0], rec_proposed[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'Proposed $d_G = \\mathbf{' + f'{loss_proposed:.3f}' + '}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

def run_training(n, m, x, train_ind, edge_index, density, tag):
    """Run GraphSAGE training and return aligned reconstruction and score."""
    K = int(np.sqrt(n) * np.log2(n) / 10)
    X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)
    eye_n = torch.eye(n)
    x_tensor = torch.FloatTensor(x)
    
    # 计算真值的标准差用于后续缩放
    x_std = np.std(x, axis=0)
    
    name = "GraphSAGE_SimpleScale"
    epochs = 200
    
    seed_everything(0)
    net = GraphSAGE_SimpleScale(m)
    optimizer = optim.Adam(net.parameters(), lr=0.002)  # 从0.001提高到0.002
    net.train()
    final_rec = None
    
    for _ in tqdm.trange(epochs, desc=f"{name}_{tag}"):
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
    
    # # 对重建结果进行标准化后再缩放到与真值相同的分布
    # rec_mean = np.mean(rec_np, axis=0, keepdims=True)
    # rec_std = np.std(rec_np, axis=0, keepdims=True) + 1e-9
    # rec_normalized = (rec_np - rec_mean) / rec_std
    # rec_scaled = rec_normalized * x_std
    
    # R, _ = orthogonal_procrustes(x, rec_scaled)
    # aligned = rec_scaled @ R.T
    
    score = float(dG(x_tensor, torch.FloatTensor(aligned)))
    
    logger.info(f"✅ {name}_{tag} training completed: dG={score:.4f}")
    return aligned, score

def main():
    """Main execution flow."""
    # n = 5000
    # m = 500
    # x, train_ind, D = prepare_data(n)
    m = 300
    x, train_ind, D, n = prepare_data()
    
    A_eball, edge_index_eball, density_eball = prepare_eball_features(D, n)
    A_knn, edge_index_knn, density_knn = prepare_knn_features(D, n)
    
    viz_results = {}
    
    aligned_eball, score_eball = run_training(n, m, x, train_ind, edge_index_eball, density_eball, "EBALL")
    viz_results["GraphSAGE_SimpleScale_EBALL"] = (aligned_eball, score_eball)
    
    aligned_knn, score_knn = run_training(n, m, x, train_ind, edge_index_knn, density_knn, "KNN")
    viz_results["GraphSAGE_SimpleScale_KNN"] = (aligned_knn, score_knn)

    
    visualize_results(x, A_eball, viz_results, n)

fig.subplots_adjust()

if not os.path.exists('visualize'):
    os.mkdir('visualize')

fig.savefig('visualize/{}_semi_moon.png'.format(datetime.datetime.now().strftime('%m%d%H%M')), bbox_inches='tight', dpi=300)
logger.info(f"✅ Figure saved: visualize/{datetime.datetime.now().strftime('%m%d%H%M')}_semi_moon.png")
# fig.savefig('imgs/%{asctime}_semi_moon.pdf', bbox_inches='tight', dpi=300)
# fig.savefig('imgs/%{asctime}_semi_moon.svg', bbox_inches='tight', dpi=300)
