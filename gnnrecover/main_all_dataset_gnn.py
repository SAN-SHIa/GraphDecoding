import sys
import os
import argparse
import csv
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from utils.datasets import get_dataset_by_name

import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.sparse import csr_matrix
import networkx as nx
import logging

from scipy.linalg import orthogonal_procrustes

import torch
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm

from gnn_utils import Net, GIN, GAT, stationary, reconstruct, dG

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Run GNN Recover on different datasets")
parser.add_argument("--dataset", type=str, default="moon", help="Dataset name to use")
args = parser.parse_args()

dataset_name = args.dataset

# Setup logging
log_file = f'logs_{dataset_name}.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

logging.info(f"===================================\nRunning dataset: {dataset_name}\n===================================")

n = 5000
m = 500
x, _ = get_dataset_by_name(dataset_name, n)
n = len(x)  # update n in case dataset returns fewer samples
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

net = GAT(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in tqdm(range(100), desc="Training GAT Model"):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_GAT = rec.detach().numpy() @ R.T
loss_GAT = float(dG(torch.FloatTensor(x), rec.detach()))
logging.info(f"GAT Loss: {loss_GAT:.4f}")

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for i in tqdm(range(100), desc="Training Proposed Model"):
    # Note 1: In the original formulation, $g$, i.e., the neural network for the scale function, should be used in reconstruct(K, pr, n, m, fr, to), namely, in the definition of $s$. We factorize $s$ and multiply g after we reconstruct the features. This is mathematically equivalent. We do this to avoid memory overflow due to long backpropagation.
    # Note 2: We roughly standardize n for stability by (n - 3000) / 3000. This does not affect the representational power of GNNs by merging them into the network parameters.
    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    rec_orig = reconstruct(K, pr, n, m, fr, to)
    rec_orig = torch.FloatTensor(rec_orig)
    g = net(torch.FloatTensor([(n - 3000) / 3000]))
    rec = rec_orig * (g ** 0.5)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_proposed = rec.detach().numpy() @ R.T
loss_proposed = float(dG(torch.FloatTensor(x), rec.detach()))
logging.info(f"Proposed Loss: {loss_proposed:.4f}")

net = GIN(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in tqdm(range(100), desc="Training GIN Model"):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_GIN = rec.detach().numpy() @ R.T
loss_GIN = float(dG(torch.FloatTensor(x), rec.detach()))
logging.info(f"GIN Loss: {loss_GIN:.4f}")

# Write results to CSV
csv_file = 'results.csv'
file_exists = os.path.isfile(csv_file)
with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['Dataset', 'Proposed Loss', 'GIN Loss', 'GAT Loss'])
    writer.writerow([dataset_name, f"{loss_proposed:.4f}", f"{loss_GIN:.4f}", f"{loss_GAT:.4f}"])
logging.info(f"Results saved to {csv_file}")

c = x[:, 0].argsort().argsort()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig = plt.figure(figsize=(20, 4))
ax = fig.add_subplot(1, 5, 1)
ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#eeeeee')
txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

visible = plt.imread('./imgs/visible.png') if os.path.exists('./imgs/visible.png') else None
if visible is not None:
    visible_ax = fig.add_axes([0.16, 0.77, 0.08, 0.1], anchor='NE', zorder=1)
    visible_ax.imshow(visible)
    visible_ax.axis('off')

G = nx.DiGraph()
G.add_edges_from([(fr[i], to[i]) for i in range(len(fr))])
ax = fig.add_subplot(1, 5, 2)
pos = nx.spring_layout(G, k=0.18, seed=0)
nx.draw_networkx(G, ax=ax, pos=pos, node_size=0.5, node_color='#005aff', labels={i: '' for i in range(n)}, edge_color='#84919e', width=0.0005, arrowsize=0.1)
txt = ax.text(0.05, 0.05, 'Input Graph', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
ax.set_rasterization_zorder(3)

ax = fig.add_subplot(1, 5, 3)
ax.scatter(rec_proposed[:, 0], rec_proposed[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
# txt = ax.text(0.05, 0.05, 'ICML2023 $d_G = \\mathbf{' + f'{loss_proposed:.3f}' + '}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt = ax.text(0.05, 0.05, 'ICML2023 $d_G = {:.3f}$'.format(loss_proposed), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(1, 5, 4)
ax.scatter(rec_GIN[:, 0], rec_GIN[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'GIN $d_G = {:.3f}$'.format(loss_GIN), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(1, 5, 5)
ax.scatter(rec_GAT[:, 0], rec_GAT[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'GAT $d_G = {:.3f}$'.format(loss_GAT), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

fig.subplots_adjust()

if not os.path.exists('imgs'):
    os.mkdir('imgs')

fig.savefig(f'imgs/semi_{dataset_name}.png', bbox_inches='tight', dpi=300)
