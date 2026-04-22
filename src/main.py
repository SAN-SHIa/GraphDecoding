import os
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import argparse
import copy
import yaml
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch_geometric.data import Data

from utils.model import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything
from utils.logging import setup_logger
from utils.datasets import get_dataset_by_name
from utils.visualization import visualize_results
import tqdm

logger = None

DEFAULT_CONFIG = {
    "task": {
        "dataset": "moon",
        "n": 5000,
        "train_ratio": 0.7
    },
    "graph": {
        "knn_k_divisor": 10.0,
        "eball_percentile": 5.0,
        "eball_scaling_factor": 2.7,
        "K": 101
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

def z_score_normalize(arr):
    """Z-score normalization for features to make neural network training stable."""
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)

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
    K = int(graph_cfg.get("K", K))

    K = max(1, K)
    A_knn, edge_index_knn = build_knn_graph(D, K)
    edge_index_knn = torch.tensor(edge_index_knn, dtype=torch.long)
    
    avg_degree = calculate_average_degree(A_knn)
    logger.info(f"🌟 KNN Graph: Nodes={n}, Edges={A_knn.nnz}, Avg Degree={avg_degree:.4f}")
    
    pr_knn = stationary(A_knn)
    pr_knn = np.maximum(pr_knn, 1e-9)
    pr_knn = np.clip(pr_knn, a_min=None, a_max=5.0)

    # Symmetrize A_knn to use the fast matrix operation
    A_knn_sym = A_knn + A_knn.T
    in_deg_knn, clust_knn = compute_symmetric_graph_features(A_knn_sym, n)

    # Normalize features
    pr_norm = z_score_normalize(pr_knn)
    in_deg_norm = z_score_normalize(in_deg_knn)
    clust_norm = z_score_normalize(clust_knn)

    density_np_knn = np.vstack([pr_norm, in_deg_norm, clust_norm]).T
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
    # Clip extreme values to prevent scaling collapse (cap at 5.0, mean is 1.0)
    pr_eball = np.clip(pr_eball, a_min=None, a_max=5.0)

    in_deg_eball, clust_eball = compute_symmetric_graph_features(A_eball, n)

    # Normalize features to mean 0, std 1 to stabilize NN scale generation
    pr_norm = z_score_normalize(pr_eball)
    in_deg_norm = z_score_normalize(in_deg_eball)
    clust_norm = z_score_normalize(clust_eball)

    density_np_eball = np.vstack([pr_norm, in_deg_norm, clust_norm]).T
    density_eball = torch.FloatTensor(density_np_eball)
    
    return A_eball, edge_index_eball, density_eball

def run_training(n, x, train_ind, edge_index, density, A, tag, training_cfg, graph_cfg):
    """Run GraphSAGE training and return aligned reconstruction and score."""

    avg_degree = int(calculate_average_degree(A))

    logger.info(f"🌟 Avg Degree={avg_degree}")

    X = torch.tensor([[avg_degree, n] for i in range(n)], dtype=torch.float)
    x_tensor = torch.FloatTensor(x)

    name = "GraphSAGE_SimpleScale"
    epochs = int(training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 0.002))

    seed_everything(0)
    net = GraphSAGE_SimpleScale()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    final_rec = None
    final_s1 = None
    final_s2 = None

    for _ in tqdm.trange(epochs, desc=f"{name}_{tag}"):
        X_with_density = torch.cat([X, density], dim=1)
        data = Data(x=X_with_density, edge_index=edge_index)
        rec, s1, s2 = net(data, return_scales=True)
        loss = dG(x_tensor[train_ind], rec[train_ind])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_rec = rec
        final_s1 = s1.detach()
        final_s2 = s2.detach()
    
    rec_np = final_rec.detach().cpu().numpy()
    R, _ = orthogonal_procrustes(x, rec_np)
    aligned = rec_np @ R.T
    
    score = float(dG(x_tensor, torch.FloatTensor(aligned)))
    
    logger.info(f"✅ {name}_{tag} training completed: dG={score:.4f}")
    return aligned, score, final_s1.numpy(), final_s2.numpy()

def visualize_scale_distribution(s1, s2, output_dir, logger):
    """Visualize scale factor distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    n_nodes, n_dims = s1.shape
    logger.info(f"📊 Scale data dimensions: {n_nodes} nodes × {n_dims} dimensions")
    
    s1_mean_per_dim = s1.mean(axis=0)
    s1_std_per_dim = s1.std(axis=0)
    s2_mean_per_dim = s2.mean(axis=0)
    s2_std_per_dim = s2.std(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    x_dims = np.arange(n_dims)
    ax1.bar(x_dims, s1_mean_per_dim, yerr=s1_std_per_dim, alpha=0.7, capsize=2, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Dimension Index', fontsize=12)
    ax1.set_ylabel('Mean Value', fontsize=12)
    ax1.set_title(f'S1: Mean ± Std per Dimension ({n_dims} dims across {n_nodes} nodes)', fontsize=14)
    ax1.set_xlim(-1, n_dims)
    
    ax2 = axes[0, 1]
    ax2.bar(x_dims, s2_mean_per_dim, yerr=s2_std_per_dim, alpha=0.7, capsize=2, color='coral')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Dimension Index', fontsize=12)
    ax2.set_ylabel('Mean Value', fontsize=12)
    ax2.set_title(f'S2: Mean ± Std per Dimension ({n_dims} dims across {n_nodes} nodes)', fontsize=14)
    ax2.set_xlim(-1, n_dims)
    
    ax3 = axes[1, 0]
    ax3.hist(s1_mean_per_dim, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=s1_mean_per_dim.mean(), color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {s1_mean_per_dim.mean():.4f}')
    ax3.set_xlabel('Mean Value', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('S1: Distribution of Dimension Means', fontsize=14)
    ax3.legend()
    
    ax4 = axes[1, 1]
    ax4.hist(s2_mean_per_dim, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(x=s2_mean_per_dim.mean(), color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {s2_mean_per_dim.mean():.4f}')
    ax4.set_xlabel('Mean Value', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('S2: Distribution of Dimension Means', fontsize=14)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_dim_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {output_dir}/scale_dim_distribution.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    s1_flat = s1.flatten()
    ax1.hist(s1_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
    ax1.axvline(x=s1_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s1_flat.mean():.4f}')
    ax1.axvline(x=np.median(s1_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(s1_flat):.4f}')
    ax1.set_xlabel('Scale Value', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'S1: All Values Distribution ({n_nodes} nodes × {n_dims} dims)', fontsize=14)
    ax1.legend()
    
    ax2 = axes[0, 1]
    s2_flat = s2.flatten()
    ax2.hist(s2_flat, bins=100, alpha=0.7, color='coral', edgecolor='none')
    ax2.axvline(x=s2_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s2_flat.mean():.4f}')
    ax2.axvline(x=np.median(s2_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(s2_flat):.4f}')
    ax2.set_xlabel('Scale Value', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'S2: All Values Distribution ({n_nodes} nodes × {n_dims} dims)', fontsize=14)
    ax2.legend()
    
    ax3 = axes[1, 0]
    s1_node_mean = s1.mean(axis=1)
    ax3.hist(s1_node_mean, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=s1_node_mean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s1_node_mean.mean():.4f}')
    ax3.set_xlabel('Mean Scale Value per Node', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('S1: Distribution of Node-wise Means', fontsize=14)
    ax3.legend()
    
    ax4 = axes[1, 1]
    s2_node_mean = s2.mean(axis=1)
    ax4.hist(s2_node_mean, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(x=s2_node_mean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {s2_node_mean.mean():.4f}')
    ax4.set_xlabel('Mean Scale Value per Node', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('S2: Distribution of Node-wise Means', fontsize=14)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_value_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {output_dir}/scale_value_distribution.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    im1 = ax1.imshow(s1.T, aspect='auto', cmap='RdBu_r', vmin=-np.abs(s1).max(), vmax=np.abs(s1).max())
    ax1.set_xlabel('Node Index', fontsize=12)
    ax1.set_ylabel('Dimension Index', fontsize=12)
    ax1.set_title('S1 Heatmap: Nodes × Dimensions', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Scale Value')
    
    ax2 = axes[1]
    im2 = ax2.imshow(s2.T, aspect='auto', cmap='RdBu_r', vmin=-np.abs(s2).max(), vmax=np.abs(s2).max())
    ax2.set_xlabel('Node Index', fontsize=12)
    ax2.set_ylabel('Dimension Index', fontsize=12)
    ax2.set_title('S2 Heatmap: Nodes × Dimensions', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='Scale Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {output_dir}/scale_heatmap.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    s1_sample = s1[np.random.choice(n_nodes, min(100, n_nodes), replace=False)]
    for i in range(min(10, n_dims)):
        ax1.plot(s1_sample[i], alpha=0.3, linewidth=0.5)
    ax1.plot(s1_sample.mean(axis=0), 'b-', linewidth=2, label='Mean')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Dimension Index', fontsize=12)
    ax1.set_ylabel('Scale Value', fontsize=12)
    ax1.set_title('S1: Sample Node Scale Vectors', fontsize=14)
    ax1.legend()
    
    ax2 = axes[1]
    s2_sample = s2[np.random.choice(n_nodes, min(100, n_nodes), replace=False)]
    for i in range(min(10, n_dims)):
        ax2.plot(s2_sample[i], alpha=0.3, linewidth=0.5)
    ax2.plot(s2_sample.mean(axis=0), 'r-', linewidth=2, label='Mean')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Dimension Index', fontsize=12)
    ax2.set_ylabel('Scale Value', fontsize=12)
    ax2.set_title('S2: Sample Node Scale Vectors', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scale_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: {output_dir}/scale_vectors.png")
    
    logger.info("="*60)
    logger.info("📊 Scale Statistics Summary")
    logger.info("="*60)
    logger.info(f"S1 Statistics:")
    logger.info(f"  - Overall Mean: {s1.mean():.6f}")
    logger.info(f"  - Overall Std: {s1.std():.6f}")
    logger.info(f"  - Min: {s1.min():.6f}")
    logger.info(f"  - Max: {s1.max():.6f}")
    logger.info(f"  - Dimension Mean Range: [{s1_mean_per_dim.min():.6f}, {s1_mean_per_dim.max():.6f}]")
    logger.info(f"  - Dimension Std Range: [{s1_std_per_dim.min():.6f}, {s1_std_per_dim.max():.6f}]")
    
    logger.info(f"S2 Statistics:")
    logger.info(f"  - Overall Mean: {s2.mean():.6f}")
    logger.info(f"  - Overall Std: {s2.std():.6f}")
    logger.info(f"  - Min: {s2.min():.6f}")
    logger.info(f"  - Max: {s2.max():.6f}")
    logger.info(f"  - Dimension Mean Range: [{s2_mean_per_dim.min():.6f}, {s2_mean_per_dim.max():.6f}]")
    logger.info(f"  - Dimension Std Range: [{s2_std_per_dim.min():.6f}, {s2_std_per_dim.max():.6f}]")
    
    logger.info(f"Scale Effect Analysis (x * (1 + s)):")
    logger.info(f"  - S1: Scale Factor Range [{1+s1.min():.4f}, {1+s1.max():.4f}]")
    logger.info(f"  - S2: Scale Factor Range [{1+s2.min():.4f}, {1+s2.max():.4f}]")
    
    s1_positive_ratio = (s1 > 0).mean() * 100
    s2_positive_ratio = (s2 > 0).mean() * 100
    logger.info(f"Positive Ratio (Scale Up):")
    logger.info(f"  - S1: {s1_positive_ratio:.2f}%")
    logger.info(f"  - S2: {s2_positive_ratio:.2f}%")

def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description="Graph Decoding on 2D datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name: moon, circles, spiral, swissroll2d, scurve2d, clusters, grid, ring, line, wave")
    parser.add_argument("--n", type=int, default=None, help="Number of samples")
    args = parser.parse_args()

    config = load_config(args.config)

    task_cfg = config.get("task", {})
    graph_cfg = config.get("graph", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    viz_cfg = config.get("visualization", {})
    
    n = args.n if args.n is not None else int(task_cfg.get("n", 5000))
    dataset_name = args.dataset if args.dataset is not None else task_cfg.get("dataset", "moon")
    train_ratio = float(task_cfg.get("train_ratio", 0.7))

    global logger
    logger = setup_logger("semi_moon", dataset_name=dataset_name)

    seed = int(runtime_cfg.get("seed", 0))
    seed_everything(seed)

    logger.info(f"🚀 Running on dataset: {dataset_name}")
    logger.info(f"📌 Config: n={n}, train_ratio={train_ratio}, seed={seed}")
    
    x, train_ind, D, n = prepare_data(n, dataset_name, train_ratio)
    
    A_eball, edge_index_eball, density_eball = prepare_eball_features(D, n, graph_cfg)
    A_knn, edge_index_knn, density_knn = prepare_knn_features(D, n, graph_cfg)
    
    viz_results = {}
    
    logger.info("🔥 start eball training...")
    aligned_eball, score_eball, s1_eball, s2_eball = run_training(n, x, train_ind, edge_index_eball, density_eball, A_eball, "EBALL", training_cfg, graph_cfg)
    viz_results["GraphSAGE_SimpleScale_EBALL"] = (aligned_eball, score_eball)
    
    scale_output_dir = os.path.join("outputs", dataset_name, "scale_analysis")
    logger.info(f"📊 Generating scale visualizations...")
    visualize_scale_distribution(s1_eball, s2_eball, scale_output_dir, logger)
    logger.info(f"✅ Scale visualizations saved to: {scale_output_dir}")

    logger.info("🔥 start knn training...")
    aligned_knn, score_knn, s1_knn, s2_knn = run_training(n, x, train_ind, edge_index_knn, density_knn, A_knn, "KNN", training_cfg, graph_cfg)
    viz_results["GraphSAGE_SimpleScale_KNN"] = (aligned_knn, score_knn)
    

    visualize_results(logger, x, A_eball, viz_results, n, dataset_name, viz_cfg)

if __name__ == "__main__":
    main()
