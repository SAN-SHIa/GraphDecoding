import os
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import argparse
import copy
import yaml

import torch
import torch.optim as optim
from torch_geometric.data import Data

from utils.model import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything
from utils.logging import setup_logger
from utils.datasets import get_dataset_by_name
from utils.visualization import visualize_results
import tqdm

logger = setup_logger("semi_moon")

DEFAULT_CONFIG = {
    "task": {
        "dataset": "moon",
        "n": 5000,
        "m": 500,
        "train_ratio": 0.7
    },
    "graph": {
        "knn_k_divisor": 10.0,
        "eball_percentile": 5.0,
        "eball_scaling_factor": 2.7
    },
    "training": {
        "epochs": 100,
        "lr": 0.002
    },
    "runtime": {
        "seed": 0
    },
    "visualization": {
        "keep_percentile": 97.0,
        "fallback_keep_percentile": 95.0,
        "min_keep_ratio": 0.5
    }
}

def deep_update(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_config(config_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must be a YAML mapping")
    return deep_update(config, loaded)

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
    # Use binary connectivity and treat graph as undirected:
    # if i->j or j->i exists, count i-j as one undirected edge.
    A_binary = A.copy()
    A_binary.data[:] = 1
    A_undirected = (A_binary + A_binary.T)
    A_undirected.data[:] = 1

    # Remove self-loops before degree calculation.
    A_undirected.setdiag(0)
    A_undirected.eliminate_zeros()

    # Degree is the number of unique neighbors in the undirected graph.
    degrees = np.array(A_undirected.sum(axis=1)).flatten()

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

def prepare_data(n, dataset_name="moon", train_ratio=0.7):
    """Prepare dataset and distance matrix."""
    if dataset_name.lower() == "moon":
        x, n = moon(n)
    else:
        x, _ = get_dataset_by_name(dataset_name, n)
        n = len(x)
    
    n_train = int(n * train_ratio)
    train_ind = torch.randperm(n)[:n_train]
    D = pairwise_distances(x)
    return x, train_ind, D, n

def prepare_knn_features(D, n, graph_cfg):
    """Build KNN graph and compute its density features."""
    knn_k_divisor = float(graph_cfg.get("knn_k_divisor", 10.0))
    K = int(np.sqrt(n) * np.log2(n) / knn_k_divisor)
    K = max(1, K)
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

def prepare_eball_features(D, n, graph_cfg):
    """Build E-ball graph and compute its density features."""
    eball_percentile = float(graph_cfg.get("eball_percentile", 5.0))
    scaling_factor = float(graph_cfg.get("eball_scaling_factor", 2.7))
    base_epsilon = np.percentile(D[D > 0], eball_percentile)
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

def run_training(n, m, x, train_ind, edge_index, density, tag, training_cfg, graph_cfg):
    """Run GraphSAGE training and return aligned reconstruction and score."""
    knn_k_divisor = float(graph_cfg.get("knn_k_divisor", 10.0))
    K = int(np.sqrt(n) * np.log2(n) / knn_k_divisor)
    K = max(1, K)
    X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)
    eye_n = torch.eye(n)
    x_tensor = torch.FloatTensor(x)
    
    name = "GraphSAGE_SimpleScale"
    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 0.002))
    
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
    parser = argparse.ArgumentParser(description="Graph Decoding on 2D datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name: moon, circles, spiral, swissroll2d, scurve2d, clusters, grid, ring, line, wave")
    parser.add_argument("--n", type=int, default=None, help="Number of samples")
    parser.add_argument("--m", type=int, default=None, help="Landmark size")
    args = parser.parse_args()

    config = load_config(args.config)

    task_cfg = config.get("task", {})
    graph_cfg = config.get("graph", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    viz_cfg = config.get("visualization", {})
    
    n = args.n if args.n is not None else int(task_cfg.get("n", 5000))
    m = args.m if args.m is not None else int(task_cfg.get("m", 500))
    dataset_name = args.dataset if args.dataset is not None else task_cfg.get("dataset", "moon")
    train_ratio = float(task_cfg.get("train_ratio", 0.7))

    seed = int(runtime_cfg.get("seed", 0))
    seed_everything(seed)
    
    logger.info(f"🚀 Running on dataset: {dataset_name}")
    logger.info(f"📌 Config: n={n}, m={m}, train_ratio={train_ratio}, seed={seed}")
    
    x, train_ind, D, n = prepare_data(n, dataset_name, train_ratio)
    
    A_eball, edge_index_eball, density_eball = prepare_eball_features(D, n, graph_cfg)
    A_knn, edge_index_knn, density_knn = prepare_knn_features(D, n, graph_cfg)
    
    viz_results = {}
    
    logger.info("🔥 start knn training...")
    aligned_knn, score_knn = run_training(n, m, x, train_ind, edge_index_knn, density_knn, "KNN", training_cfg, graph_cfg)
    viz_results["GraphSAGE_SimpleScale_KNN"] = (aligned_knn, score_knn)
    
    logger.info("🔥 start eball training...")
    aligned_eball, score_eball = run_training(n, m, x, train_ind, edge_index_eball, density_eball, "EBALL", training_cfg, graph_cfg)
    viz_results["GraphSAGE_SimpleScale_EBALL"] = (aligned_eball, score_eball)
    
    visualize_results(logger, x, A_eball, viz_results, n, dataset_name, viz_cfg)

if __name__ == "__main__":
    main()
