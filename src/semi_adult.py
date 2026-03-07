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

from util import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything, setup_logger
import tqdm
import datetime

logger = setup_logger("semi_moon")
seed_everything(0)

def build_knn_graph(D, K):
    """Construct KNN graph"""
    n = D.shape[0]
    fr = np.arange(n).repeat(K).reshape(-1)
    to = np.argsort(D, axis=1)[:, 1:K + 1].reshape(-1)
    A = csr_matrix((np.ones(n * K) / K, (fr, to)))
    edge_index = np.vstack([fr, to])
    return A, edge_index

def build_eball_graph(D, epsilon):
    """Construct ε-ball graph"""
    n = D.shape[0]
    # Create adjacency matrix
    adj = (D < epsilon) & (D > 0)  # Exclude self-connections
    fr, to = np.where(adj)
    
    # Compute edge weights (using inverse distance)
    distances = D[fr, to]
    weights = 1.0 / (distances + 1e-9)  # Avoid division by zero
    
    if len(fr) > 0:
        A = csr_matrix((weights, (fr, to)), shape=(n, n))
    else:
        A = csr_matrix((n, n))
    
    edge_index = np.vstack([fr, to]) if len(fr) > 0 else np.empty((2, 0))
    return A, edge_index

def calculate_average_degree(A):
    """Calculate the average degree of each node in the graph"""
    # Create binary matrix, only care about connection existence
    A_binary = A.copy()
    A_binary.data[:] = 1
    
    # Remove self-loops
    A_binary.setdiag(0)
    A_binary.eliminate_zeros()
    
    # Calculate degree (number of neighbors) for each node
    degrees = np.array(A_binary.sum(axis=0)).flatten()
    
    # Return average degree
    return np.mean(degrees) if len(degrees) > 0 else 0

def compute_symmetric_graph_features(A, n):
    """
    Accelerate computation of symmetric graph features (In-degree, Clustering Coefficient) using matrix operations
    Replaces NetworkX to improve speed on large graphs
    """
    # 1. Compute degree (In-degree = Out-degree for symmetric graphs)
    # A contains weights, but use binary connections for topological feature calculation
    A_bool = (A > 0)
    d = np.array(A_bool.sum(axis=1)).flatten()
    in_deg = d / (np.mean(d) + 1e-9)

    # 2. Compute clustering coefficient
    # For undirected (symmetric) graphs: C_i = (A^3)_ii / (d_i * (d_i - 1))
    # Use dense matrix to compute diagonal of A^3 (N=5000 ~100MB RAM, very fast)
    A_dense = A_bool.astype(np.float32).toarray()
    
    # Compute A^2
    A2 = A_dense @ A_dense
    
    # Compute (A^3)_ii = sum_j (A^2)_ij * A_ji = sum_j (A^2)_ij * A_ij (since symmetric)
    # Efficiently compute diagonal using einsum
    diag_A3 = np.einsum('ij,ij->i', A2, A_dense)
    
    # Compute clustering coefficient
    denom = d * (d - 1)
    clust = np.zeros_like(d, dtype=np.float32)
    mask = denom > 0
    clust[mask] = diag_A3[mask] / denom[mask]
    
    clust = clust / (np.mean(clust) + 1e-9)
    
    return in_deg, clust

def visualize_results(x, A_eball, viz_results, n):
    logger.info("🎨 visualization...")
    
    # Extract results safely
    rec_simple_eball, loss_simple_eball = viz_results.get("GraphSAGE_SimpleScale_EBALL", (None, 0))
    rec_simple_knn, loss_simple_knn = viz_results.get("GraphSAGE_SimpleScale_KNN", (None, 0))

    c = x[:, 0].argsort().argsort()
    fig = plt.figure(figsize=(14, 8))

    # 1. Ground Truth
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#eeeeee')
    txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])
    
    if os.path.exists('./imgs/visible.png'):
        visible = plt.imread('./imgs/visible.png')
        visible_ax = fig.add_axes([0.24, 0.77, 0.1, 0.1], anchor='NE', zorder=1)
        visible_ax.imshow(visible)
        visible_ax.axis('off')

    # 2. Input Graph (E-ball)
    fr, to = A_eball.nonzero()
    G = nx.DiGraph()
    G.add_edges_from(zip(fr, to))

    ax = fig.add_subplot(2, 2, 2)
    pos = {i: x[i] for i in range(n)}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=0.5, node_color='#005aff')
    
    if G.number_of_edges() > 2000:
        edges_to_draw = list(G.edges())[:2000]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_to_draw, edge_color='#84919e', width=0.0005, arrowsize=0.1)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#84919e', width=0.0005, arrowsize=0.1)
    
    txt = ax.text(0.05, 0.05, 'Input Graph (Real Pos)', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    ax.set_rasterization_zorder(3)
    ax.axis('off')

    # 3. SimpleScale (E-ball)
    if rec_simple_eball is not None:
        ax = fig.add_subplot(2, 2, 3)
        ax.scatter(rec_simple_eball[:, 0], rec_simple_eball[:, 1], c=c, s=10, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = ax.text(0.05, 0.05, f'SimpleScale (E-ball) $d_G = {loss_simple_eball:.2f}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    # 4. SimpleScale (KNN)
    if rec_simple_knn is not None:
        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(rec_simple_knn[:, 0], rec_simple_knn[:, 1], c=c, s=10, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = ax.text(0.05, 0.05, f'SimpleScale (KNN) $d_G = {loss_simple_knn:.2f}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

    fig.subplots_adjust()

    if not os.path.exists('visualize'):
        os.mkdir('visualize')

    save_path = 'visualize/{}_semi_moon.png'.format(datetime.datetime.now().strftime('%m%d%H%M'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"📂 Figure saved: {save_path}")

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

def prepare_knn_features(D, n):
    """Build KNN graph and compute its density features."""
    K = int(np.sqrt(n) * np.log2(n) / 10)
    A_knn, edge_index_knn = build_knn_graph(D, K)
    edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long)
    
    avg_degree = calculate_average_degree(A_knn)
    logger.info(f"🌟 KNN Graph: Nodes={n}, Edges={A_knn.nnz}, Avg Degree={avg_degree:.4f}")
    
    pr_knn = stationary(A_knn)
    pr_knn = np.maximum(pr_knn, 1e-9)

    G_nx_knn = nx.from_scipy_sparse_array(A_knn, create_using=nx.DiGraph)
    in_deg_knn = np.array([G_nx_knn.in_degree(i) for i in range(n)])
    in_deg_knn = in_deg_knn / (np.mean(in_deg_knn) + 1e-9)

    clust_knn = nx.clustering(G_nx_knn)
    clust_knn = np.array([clust_knn[i] for i in range(n)])
    clust_knn = clust_knn / (np.mean(clust_knn) + 1e-9)

    density_np_knn = np.vstack([pr_knn, in_deg_knn, clust_knn]).T
    density_knn = torch.FloatTensor(density_np_knn)
    
    return A_knn, edge_index_knn, density_knn

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

    in_deg_eball, clust_eball = compute_symmetric_graph_features(A_eball, n)

    density_np_eball = np.vstack([pr_eball, in_deg_eball, clust_eball]).T
    density_eball = torch.FloatTensor(density_np_eball)
    
    return A_eball, edge_index_eball, density_eball

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

if __name__ == "__main__":
    main()
