import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
import argparse
import tqdm
import json
from datetime import datetime

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import main as main_module
from main import load_config, prepare_data, prepare_eball_features, run_training, calculate_average_degree
from utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="moon")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--min_p", type=float, default=0.1)
    parser.add_argument("--max_p", type=float, default=30.0)
    parser.add_argument("--step_p", type=float, default=0.1)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    graph_cfg = config.get("graph", {})
    training_cfg = config.get("training", {})
    
    # Force scaling factor to 1.0 as requested
    graph_cfg["eball_scaling_factor"] = 1.0
    
    logger = setup_logger("percolation_test", dataset_name=args.dataset)
    main_module.logger = logger  # Inject logger into main module
    
    # Prepare data once
    x, train_ind, D, n = prepare_data(args.n, args.dataset, 0.7)
    
    percentiles = np.arange(args.min_p, args.max_p + 1e-5, args.step_p)
    
    results_lcc = []
    results_dg = []
    results_deg = []
    
    # Setup JSON log file
    output_dir = os.path.join("outputs", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_log_path = os.path.join(output_dir, f"percolation_log_{timestamp}.json")
    
    logger.info(f"Starting percolation test: {len(percentiles)} points from {args.min_p} to {args.max_p}")
    logger.info(f"Logging results in real-time to: {json_log_path}")
    
    # Initialize the JSON file with an empty list
    with open(json_log_path, 'w') as f:
        json.dump([], f)
        
    for p in tqdm.tqdm(percentiles, desc="Testing percentiles"):
        graph_cfg["eball_percentile"] = float(p)
        
        # We need epsilon, calculate it here or fetch it from prepare_eball_features
        # It's calculated inside prepare_eball_features but not returned. Let's calculate it here.
        base_epsilon = np.percentile(D[D > 0], float(p))
        epsilon = base_epsilon * graph_cfg.get("eball_scaling_factor", 1.0)
        
        # 1. Build graph and features
        A_eball, edge_index_eball, density_eball = prepare_eball_features(D, n, graph_cfg)
        
        # 2. Compute LCC ratio
        if A_eball.nnz > 0:
            n_components, labels = connected_components(csgraph=A_eball, directed=False, return_labels=True)
            unique, counts = np.unique(labels, return_counts=True)
            lcc_ratio = counts.max() / n
        else:
            lcc_ratio = 1.0 / n
            
        # 3. Compute avg degree
        avg_degree = calculate_average_degree(A_eball)
        
        # 4. Train model and get dG score
        _, dG_score, _, _ = run_training(n, x, train_ind, edge_index_eball, density_eball, A_eball, f"P_{p:.1f}", training_cfg, graph_cfg)
        
        results_lcc.append(lcc_ratio)
        results_dg.append(dG_score)
        results_deg.append(avg_degree)
        
        logger.info(f"p={p:.2f} | epsilon={epsilon:.6f} | LCC={lcc_ratio:.8f} | AvgDeg={avg_degree:.4f} | dG={dG_score:.4f}")
        
        # Real-time JSON logging
        record = {
            "percentile_p": float(p),
            "epsilon": round(float(epsilon), 6),
            "E_ball_Graph": {
                "Nodes": int(n),
                "Edges": int(A_eball.nnz),
                "Avg_Degree": float(avg_degree)
            },
            "metrics": {
                "LCC_Ratio": round(float(lcc_ratio), 8),
                "dG_Score": float(dG_score)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to JSON file by reading, appending, and rewriting
        with open(json_log_path, 'r') as f:
            data = json.load(f)
        data.append(record)
        with open(json_log_path, 'w') as f:
            json.dump(data, f, indent=4)

    # Plotting
    # Set global font to Times New Roman as per user preference
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = '#005aff' # User preferred blue
    ax1.set_xlabel('e-ball Percentile (%)', fontsize=14)
    ax1.set_ylabel('LCC Ratio (Connectivity)', color=color, fontsize=14)
    ax1.plot(percentiles, results_lcc, marker='o', color=color, label='LCC Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color_dg = '#d62728' # Red for dG
    ax2.set_ylabel('dG Score (Reconstruction Error)', color=color_dg, fontsize=14)  
    ax2.plot(percentiles, results_dg, marker='s', color=color_dg, label='dG Score')
    ax2.tick_params(axis='y', labelcolor=color_dg)

    fig.tight_layout()  
    
    out_path = os.path.join(output_dir, 'percolation_analysis.png')
    plt.title(f'Percolation Phase Transition (scaling_factor=1.0)', fontsize=16)
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    # Also plot against Average Degree
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Average Degree (dG)', fontsize=14)
    ax1.set_ylabel('LCC Ratio (Connectivity)', color=color, fontsize=14)
    ax1.plot(results_deg, results_lcc, marker='o', color=color, label='LCC Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    ax2.set_ylabel('dG Score (Reconstruction Error)', color=color_dg, fontsize=14)  
    ax2.plot(results_deg, results_dg, marker='s', color=color_dg, label='dG Score')
    ax2.tick_params(axis='y', labelcolor=color_dg)

    fig.tight_layout()  
    out_path_deg = os.path.join(output_dir, 'percolation_analysis_degree.png')
    plt.title(f'Percolation Phase Transition vs Avg Degree', fontsize=16)
    plt.savefig(out_path_deg, dpi=300)
    plt.close()

    logger.info(f"Done! Plots saved to {out_path} and {out_path_deg}")

if __name__ == "__main__":
    main()
