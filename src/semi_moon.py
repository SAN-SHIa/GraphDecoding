import os
import argparse
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
import yaml

logger = setup_logger("semi_moon")


DEFAULT_CONFIG_PATH = os.path.join("configs", "semi_moon.yaml")


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected a mapping.")

    required_sections = ["data", "graph", "training", "visualization"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in {config_path}.")

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run semi_moon experiment with external config.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config file")
    return parser.parse_args()

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

def visualize_results(x, A_eball, viz_results, n, config):
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
    
    max_edges_to_draw = int(config["visualization"]["max_edges_to_draw"])
    if G.number_of_edges() > max_edges_to_draw:
        edges_to_draw = list(G.edges())[:max_edges_to_draw]
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

    output_dir = config["visualization"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, '{}_semi_moon.png'.format(datetime.datetime.now().strftime('%m%d%H%M')))
    fig.savefig(save_path, bbox_inches='tight', dpi=int(config["visualization"]["dpi"]))
    logger.info(f"📂 Figure saved: {save_path}")

def prepare_data(n, train_ratio):
    """Prepare moon dataset and distance matrix."""
    x, n = moon(n)
    n_train = int(n * train_ratio)
    train_ind = torch.randperm(n)[:n_train]
    D = pairwise_distances(x)
    return x, train_ind, D

def prepare_knn_features(D, n, knn_divisor):
    """Build KNN graph and compute its density features."""
    K = int(np.sqrt(n) * np.log2(n) / knn_divisor)
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
    
    return A_knn, edge_index_knn, density_knn, K

def prepare_eball_features(D, n, epsilon_percentile, scaling_factor):
    """Build E-ball graph and compute its density features."""
    base_epsilon = np.percentile(D[D > 0], epsilon_percentile)
    epsilon = base_epsilon * scaling_factor
    logger.info(f"Using scaling factor={scaling_factor}, percen epsilon={epsilon:.6f}")
    
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

def run_training(n, m, x, train_ind, edge_index, density, tag, feature_k, epochs, lr):
    """Run GraphSAGE training and return aligned reconstruction and score."""
    X = torch.tensor([[feature_k, n] for i in range(n)], dtype=torch.float)
    eye_n = torch.eye(n)
    x_tensor = torch.FloatTensor(x)
    
    name = "GraphSAGE_SimpleScale"
    
    seed_everything(0)
    net = GraphSAGE_SimpleScale(m)
    optimizer = optim.Adam(net.parameters(), lr=lr)
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
    score = float(dG(x_tensor, torch.FloatTensor(aligned)))
    
    logger.info(f"✅ {name}_{tag} training completed: dG={score:.4f}")
    return aligned, score

def main():
    """Main execution flow."""
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 0))
    seed_everything(seed)

    n = int(config["data"]["n"])
    train_ratio = float(config["data"]["train_ratio"])
    m = int(config["training"]["m"])
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["lr"])
    knn_divisor = float(config["graph"]["knn"]["divisor"])
    epsilon_percentile = float(config["graph"]["eball"]["epsilon_percentile"])
    scaling_factor = float(config["graph"]["eball"]["scaling_factor"])

    x, train_ind, D = prepare_data(n, train_ratio)
    
    A_eball, edge_index_eball, density_eball = prepare_eball_features(D, n, epsilon_percentile, scaling_factor)
    A_knn, edge_index_knn, density_knn, feature_k = prepare_knn_features(D, n, knn_divisor)
    
    viz_results = {}
    
    aligned_knn, score_knn = run_training(n, m, x, train_ind, edge_index_knn, density_knn, "KNN", feature_k, epochs, lr)
    viz_results["GraphSAGE_SimpleScale_KNN"] = (aligned_knn, score_knn)
    aligned_eball, score_eball = run_training(n, m, x, train_ind, edge_index_eball, density_eball, "EBALL", feature_k, epochs, lr)
    viz_results["GraphSAGE_SimpleScale_EBALL"] = (aligned_eball, score_eball)
    
    visualize_results(x, A_eball, viz_results, n, config)

if __name__ == "__main__":
    main()
