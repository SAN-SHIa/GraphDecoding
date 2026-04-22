"""
分析比较不同的流形学图特征参数，选出最有价值的特征组合用于神经网络
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import trustworthiness
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import sys
import os
import matplotlib.font_manager as fm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# The Linux environment doesn't have Times New Roman TTF installed, 
# and external downloads are blocked by proxy.
# We will set the font to Times New Roman so that it tries to use it,
# but we suppress the matplotlib findfont warnings so it silently falls back 
# to DejaVu Serif when rendering PNGs.
plt_font_name = 'Times New Roman'

import torch
import torch.optim as optim
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything
from utils.logging import setup_logger
from utils.datasets import get_dataset_by_name
import tqdm
import argparse
import datetime

logger = setup_logger("feature_analysis")
seed_everything(0)

def build_eball_graph(D, epsilon):
    """构建 ε-ball 图"""
    n = D.shape[0]
    adj = (D < epsilon) & (D > 0)
    fr, to = np.where(adj)
    distances = D[fr, to]
    weights = 1.0 / (distances + 1e-9)

    if len(fr) > 0:
        A = csr_matrix((weights, (fr, to)), shape=(n, n))
    else:
        A = csr_matrix((n, n))

    edge_index = np.vstack([fr, to]) if len(fr) > 0 else np.empty((2, 0), dtype=np.int64)
    return A, edge_index

def calculate_average_degree(A):
    """计算平均度数，用于描述图稠密程度"""
    degree = np.array((A > 0).sum(axis=1)).flatten()
    return float(degree.mean()) if degree.size > 0 else 0.0

def safe_corr(metric_fn, a, b):
    """常数向量时相关系数会失败，这里统一兜底到 0"""
    try:
        value, _ = metric_fn(a, b)
        return float(value) if np.isfinite(value) else 0.0
    except Exception:
        return 0.0

def prepare_graph_features(D, percentile, scaling_factor):
    """按照给定参数构建图并返回图特征与统计信息"""
    positive_distances = D[D > 0]
    base_epsilon = np.percentile(positive_distances, percentile)
    epsilon = float(base_epsilon * scaling_factor)
    A, edge_index = build_eball_graph(D, epsilon)
    features = compute_all_features(A, D, D.shape[0])

    graph_stats = {
        'percentile': float(percentile),
        'scaling_factor': float(scaling_factor),
        'base_epsilon': float(base_epsilon),
        'epsilon': epsilon,
        'edge_count': int(A.nnz),
        'avg_degree': calculate_average_degree(A),
    }
    return A, torch.tensor(edge_index, dtype=torch.long), features, graph_stats

def compute_all_features(A, D, n):
    """
    计算所有候选特征
    返回特征字典：{特征名: 特征向量}
    """
    features = {}

    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    features['stationary distribution'] = pr / (np.mean(pr) + 1e-9)

    A_bool = (A > 0)
    degree = np.array(A_bool.sum(axis=1)).flatten()
    features['degree'] = degree / (np.mean(degree) + 1e-9)

    A_dense = A_bool.astype(np.float32).toarray()
    A2 = A_dense @ A_dense
    diag_A3 = np.einsum('ij,ij->i', A2, A_dense)
    denom = degree * (degree - 1)
    clust = np.zeros_like(degree, dtype=np.float32)
    mask = denom > 0
    clust[mask] = diag_A3[mask] / denom[mask]
    features['clustering coefficient'] = clust / (np.mean(clust) + 1e-9)

    k = min(10, n - 1)
    sorted_D = np.sort(D, axis=1)
    local_density = 1.0 / (np.mean(sorted_D[:, 1:k + 1], axis=1) + 1e-9)
    features['local density'] = local_density / (np.mean(local_density) + 1e-9)

    knn_std = np.std(sorted_D[:, 1:k + 1], axis=1)
    features['knn variance'] = knn_std / (np.mean(knn_std) + 1e-9)

    A2_bool = (A2 > 0)
    two_hop = np.array(A2_bool.sum(axis=1)).flatten()
    features['two hop neighbors'] = two_hop / (np.mean(two_hop) + 1e-9)

    neighbor_deg_sum = A_bool @ degree
    avg_neighbor_deg = np.zeros(n)
    mask = degree > 0
    avg_neighbor_deg[mask] = neighbor_deg_sum[mask] / degree[mask]
    features['avg neighbor degree'] = avg_neighbor_deg / (np.mean(avg_neighbor_deg) + 1e-9)

    G_nx = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    try:
        core_number = nx.core_number(G_nx)
        core = np.array([core_number.get(i, 0) for i in range(n)])
        features['core number'] = core / (np.mean(core) + 1e-9)
    except Exception:
        features['core number'] = np.ones(n)

    try:
        if n > 500:
            betweenness = nx.betweenness_centrality(G_nx, k=min(100, n), seed=0)
        else:
            betweenness = nx.betweenness_centrality(G_nx)
        bet = np.array([betweenness.get(i, 0) for i in range(n)])
        features['betweenness'] = bet / (np.mean(bet) + 1e-9)
    except Exception:
        features['betweenness'] = np.ones(n)

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_nx)
        eig = np.array([eigenvector.get(i, 0) for i in range(n)])
        features['eigenvector'] = eig / (np.mean(eig) + 1e-9)
    except Exception:
        features['eigenvector'] = np.ones(n)

    try:
        closeness = nx.closeness_centrality(G_nx)
        close = np.array([closeness.get(i, 0) for i in range(n)])
        features['closeness'] = close / (np.mean(close) + 1e-9)
    except Exception:
        features['closeness'] = np.ones(n)

    weight_sum = np.array(A.sum(axis=1)).flatten()
    features['weight sum'] = weight_sum / (np.mean(weight_sum) + 1e-9)

    for name in features:
        features[name] = np.nan_to_num(features[name], nan=1.0, posinf=1.0, neginf=1.0)

    return features

def evaluate_feature_importance(features, x):
    """
    评估每个特征与真实坐标的相关性

    方法:
    1. Pearson 相关系数
    2. Spearman 相关系数
    3. 互信息
    """
    results = {}
    x_ref = x[:, :min(2, x.shape[1])]

    for name, feat in features.items():
        feat = feat.reshape(-1)
        pearson_scores = []
        spearman_scores = []
        mi_scores = []

        for dim in range(x_ref.shape[1]):
            target = x_ref[:, dim]
            pearson_scores.append(abs(safe_corr(pearsonr, feat, target)))
            spearman_scores.append(abs(safe_corr(spearmanr, feat, target)))
            try:
                mi_value = mutual_info_regression(feat.reshape(-1, 1), target, random_state=0)[0]
            except Exception:
                mi_value = 0.0
            mi_scores.append(float(mi_value))

        results[name] = {
            'max_pearson': max(pearson_scores),
            'max_spearman': max(spearman_scores),
            'max_mi': max(mi_scores),
            'combined_score': float(np.mean([
                max(pearson_scores),
                max(spearman_scores),
                max(mi_scores)
            ]))
        }

    return results

def evaluate_manifold_metrics(x_true, x_pred, k=5):
    """使用全局与局部指标共同评估流形恢复质量"""
    x_true = np.asarray(x_true, dtype=np.float32)
    x_pred = np.asarray(x_pred, dtype=np.float32)

    R, _ = orthogonal_procrustes(x_true, x_pred)
    x_aligned = x_pred @ R.T

    x_true_tensor = torch.FloatTensor(x_true)
    x_aligned_tensor = torch.FloatTensor(x_aligned)
    dg_value = float(dG(x_true_tensor, x_aligned_tensor))
    aligned_error = float(np.mean(np.linalg.norm(x_true - x_aligned, axis=1)))

    k = max(2, min(int(k), len(x_true) - 1))
    trust = float(trustworthiness(x_true, x_aligned, n_neighbors=k))
    cont = float(trustworthiness(x_aligned, x_true, n_neighbors=k))

    return {
        'dG': dg_value,
        'aligned_error': aligned_error,
        'trustworthiness': trust,
        'continuity': cont,
        'k_neighbors': k,
        'aligned': x_aligned,
    }

def build_feature_matrix(features, combo):
    """将特征组合转换成标准化后的 3 维矩阵"""
    feature_matrix = np.hstack([features[name].reshape(-1, 1) for name in combo])
    return StandardScaler().fit_transform(feature_matrix)

def rank_metric_dataframe(df, smaller_is_better, larger_is_better, rank_col='total_rank_score'):
    """用 Borda Count 融合多项指标排名"""
    ranked_df = df.copy()

    for col in smaller_is_better:
        ranked_df[f'rank_{col}'] = ranked_df[col].rank(method='min', ascending=True)
    for col in larger_is_better:
        ranked_df[f'rank_{col}'] = ranked_df[col].rank(method='min', ascending=False)

    rank_columns = [f'rank_{col}' for col in smaller_is_better + larger_is_better]
    ranked_df[rank_col] = ranked_df[rank_columns].sum(axis=1)
    ranked_df = ranked_df.sort_values(
        by=[rank_col] + smaller_is_better + larger_is_better,
        ascending=[True] + [True] * len(smaller_is_better) + [False] * len(larger_is_better)
    ).reset_index(drop=True)
    return ranked_df

def train_with_features(x, edge_index, feature_matrix, n, avg_degree, epochs=50):
    """使用给定特征矩阵训练模型并返回多维评价指标"""
    if feature_matrix.shape[1] != 3:
        raise ValueError(
            f"GraphSAGE_SimpleScale 需要 3 维图特征，当前收到 {feature_matrix.shape[1]} 维"
        )

    seed_everything(0)
    X = torch.tensor([[avg_degree, n] for _ in range(n)], dtype=torch.float)
    x_tensor = torch.FloatTensor(x)
    density = torch.FloatTensor(feature_matrix)

    n_train = int(n * 0.7)
    train_ind = torch.randperm(n)[:n_train]

    net = GraphSAGE_SimpleScale()
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    net.train()

    for _ in range(epochs):
        data = Data(x=torch.cat([X, density], dim=1), edge_index=edge_index)
        rec = net(data)
        loss = dG(x_tensor[train_ind], rec[train_ind])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        data = Data(x=torch.cat([X, density], dim=1), edge_index=edge_index)
        rec_np = net(data).cpu().numpy()

    metrics = evaluate_manifold_metrics(x, rec_np, k=max(5, int(n * 0.01)))
    return metrics

def run_combination_experiment_scientific(features, feature_names, x, edge_index, n, avg_degree,
                                          num_features=3, epochs=30):
    """使用 dG + trustworthiness + continuity 对特征组合进行综合排名"""
    results = []
    all_combinations = list(combinations(feature_names, num_features))
    logger.info(f"📊 测试 {len(all_combinations)} 种特征组合...")

    for combo in tqdm.tqdm(all_combinations, desc="Testing combinations"):
        feature_matrix = build_feature_matrix(features, combo)
        metrics = train_with_features(
            x, edge_index, feature_matrix, n, avg_degree, epochs=epochs
        )
        results.append({
            'combination': combo,
            'dG': metrics['dG'],
            'aligned_error': metrics['aligned_error'],
            'trustworthiness': metrics['trustworthiness'],
            'continuity': metrics['continuity'],
            'k_neighbors': metrics['k_neighbors'],
        })

    df = pd.DataFrame(results)
    return rank_metric_dataframe(
        df,
        smaller_is_better=['dG', 'aligned_error'],
        larger_is_better=['trustworthiness', 'continuity']
    )

def run_parameter_search_moon(x, D, n, percentiles, scaling_factors, top_feature_count=6,
                              num_features=3, epochs=30):
    """在 moon 数据集上对不同图参数进行综合打分，作为最终选择标准"""
    search_results = []

    for percentile in percentiles:
        for scaling_factor in scaling_factors:
            A, edge_index, features, graph_stats = prepare_graph_features(D, percentile, scaling_factor)
            logger.info(
                f"🔍 参数评估 percentile={percentile:.2f}, scale={scaling_factor:.2f}, "
                f"epsilon={graph_stats['epsilon']:.6f}, avg_degree={graph_stats['avg_degree']:.2f}"
            )

            if graph_stats['edge_count'] == 0:
                logger.warning("图中没有边，跳过该参数组合")
                continue

            importance = evaluate_feature_importance(features, x)
            sorted_features = sorted(
                importance.items(), key=lambda item: item[1]['combined_score'], reverse=True
            )
            shortlist_size = max(num_features, top_feature_count)
            candidate_features = [name for name, _ in sorted_features[:shortlist_size]]

            combo_df = run_combination_experiment_scientific(
                features,
                candidate_features,
                x,
                edge_index,
                n,
                graph_stats['avg_degree'],
                num_features=num_features,
                epochs=epochs,
            )
            best_row = combo_df.iloc[0]

            search_results.append({
                'percentile': float(percentile),
                'scaling_factor': float(scaling_factor),
                'epsilon': graph_stats['epsilon'],
                'edge_count': graph_stats['edge_count'],
                'avg_degree': graph_stats['avg_degree'],
                'candidate_features': tuple(candidate_features),
                'best_combination': tuple(best_row['combination']),
                'dG': float(best_row['dG']),
                'aligned_error': float(best_row['aligned_error']),
                'trustworthiness': float(best_row['trustworthiness']),
                'continuity': float(best_row['continuity']),
                'best_combo_rank_score': float(best_row['total_rank_score']),
            })

    if not search_results:
        return pd.DataFrame()

    param_df = pd.DataFrame(search_results)
    return rank_metric_dataframe(
        param_df,
        smaller_is_better=['dG', 'aligned_error', 'best_combo_rank_score'],
        larger_is_better=['trustworthiness', 'continuity']
    )

def visualize_feature_correlation(features, dataset_name="moon", tag="default"):
    """
    可视化不同特征之间的相关性热力图

    Parameters:
    -----------
    features : dict
        特征字典 {特征名: 特征向量}
    dataset_name : str
        数据集名称，用于保存文件
    tag : str
        结果标签，便于区分不同参数设置
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [plt_font_name]

    feature_names = list(features.keys())
    n_features = len(feature_names)

    feature_matrix = np.column_stack([features[name] for name in feature_names])

    pearson_corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            pearson_corr[i, j] = safe_corr(pearsonr, feature_matrix[:, i], feature_matrix[:, j])

    fig, ax1 = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pearson_corr,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax1,
        annot_kws={'size': 10, 'family': plt_font_name},
    )
    ax1.set_title('Pearson Correlation', fontsize=16, fontweight='bold', fontfamily=plt_font_name)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=14, fontfamily=plt_font_name)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=14, fontfamily=plt_font_name)

    plt.tight_layout()
    os.makedirs('visualize', exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    save_path = f'visualize/feature_correlation_{dataset_name}_{tag}_{timestamp}.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"📊 特征相关性热力图已保存: {save_path}")
    plt.close(fig)

    corr_msg = []
    corr_msg.append("\n" + "=" * 60)
    corr_msg.append("高度相关的特征对 (|Pearson| > 0.8):")
    corr_msg.append("=" * 60)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(pearson_corr[i, j]) > 0.8:
                corr_msg.append(f"  {feature_names[i]} <-> {feature_names[j]}: r = {pearson_corr[i, j]:.3f}")
    logger.info("\n".join(corr_msg))

    return pearson_corr

def print_importance_table(sorted_features):
    output = []
    output.append("\n" + "=" * 80)
    output.append("特征重要性排名 (综合得分 = Pearson + Spearman + 互信息)")
    output.append("=" * 80)
    output.append(f"{'排名':<4} {'特征名':<22} {'Pearson':<10} {'Spearman':<10} {'互信息':<10} {'综合得分':<10}")
    output.append("-" * 80)
    for rank, (name, scores) in enumerate(sorted_features, 1):
        output.append(
            f"{rank:<4} {name:<22} {scores['max_pearson']:<10.4f} "
            f"{scores['max_spearman']:<10.4f} {scores['max_mi']:<10.4f} {scores['combined_score']:<10.4f}"
        )
    return "\n".join(output)

def print_combination_table(df, title, top_k=10):
    output = []
    output.append("\n" + "=" * 100)
    output.append(title)
    output.append("=" * 100)
    columns = ['combination', 'dG', 'aligned_error', 'trustworthiness', 'continuity', 'total_rank_score']
    output.append(df[columns].head(top_k).to_string(index=False))
    return "\n".join(output)

def print_parameter_table(df, title, top_k=10):
    output = []
    output.append("\n" + "=" * 120)
    output.append(title)
    output.append("=" * 120)
    columns = [
        'percentile', 'scaling_factor', 'epsilon', 'avg_degree', 'best_combination',
        'dG', 'aligned_error', 'trustworthiness', 'continuity', 'total_rank_score'
    ]
    output.append(df[columns].head(top_k).to_string(index=False))
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Feature Analysis for Graph Decoding")
    parser.add_argument("--dataset", type=str, default="moon", help="Dataset name")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--m", type=int, default=100, help="Retained only for compatibility")
    parser.add_argument("--combo_size", type=int, default=3, help="Number of features per combination")
    parser.add_argument("--percentile", type=float, default=5.0, help="Baseline epsilon percentile")
    parser.add_argument("--scale", type=float, default=2.7, help="Baseline epsilon scaling factor")
    parser.add_argument("--search_percentiles", type=float, nargs='+', default=[3.0, 5.0, 7.0], help="Moon 参数搜索的 percentile 网格")
    parser.add_argument("--search_scales", type=float, nargs='+', default=[2.0, 2.7, 3.5], help="Moon 参数搜索的 scaling 网格")
    parser.add_argument("--search_top_features", type=int, default=6, help="参数搜索时先保留的高分特征数")
    parser.add_argument("--search_epochs", type=int, default=30, help="组合和参数评分时的训练轮数")
    parser.add_argument("--full_search", action="store_true", help="对当前图参数下的全部特征组合做综合搜索")
    parser.add_argument("--skip_param_search", action="store_true", help="跳过 moon 参数搜索")
    parser.add_argument("--skip_visualization", action="store_true", help="跳过特征相关性热力图")
    args = parser.parse_args()

    n = args.n
    dataset_name = args.dataset

    logger.info(f"🚀 特征分析: 数据集={dataset_name}, n={n}")

    if dataset_name.lower() == "moon":
        x, n = moon(n)
    else:
        x, _ = get_dataset_by_name(dataset_name, n)
        n = len(x)

    D = pairwise_distances(x)

    A, edge_index, features, graph_stats = prepare_graph_features(D, args.percentile, args.scale)
    logger.info(
        f"📈 基线图构建完成: 节点={n}, 边={graph_stats['edge_count']}, "
        f"avg_degree={graph_stats['avg_degree']:.2f}, epsilon={graph_stats['epsilon']:.6f}"
    )

    if not args.skip_visualization:
        logger.info("🎨 生成基线特征相关性热力图...")
        visualize_feature_correlation(features, dataset_name, tag="baseline")

    logger.info("📊 评估特征重要性...")
    importance = evaluate_feature_importance(features, x)
    sorted_features = sorted(importance.items(), key=lambda item: item[1]['combined_score'], reverse=True)
    feature_names = [name for name, _ in sorted_features]
    logger.info(print_importance_table(sorted_features))

    top_features = [name for name, _ in sorted_features[:max(args.combo_size, args.search_top_features)]]
    top_combo_seed = top_features[:args.combo_size]
    
    recom_msg = []
    recom_msg.append("\n" + "=" * 80)
    recom_msg.append(f"📌 基线参数下推荐优先探索的特征: {top_features}")
    recom_msg.append(f"📌 基线参数下相关性最强的前 {args.combo_size} 个特征: {top_combo_seed}")
    recom_msg.append("=" * 80)
    logger.info("\n".join(recom_msg))

    if args.full_search:
        combo_feature_names = feature_names
        combo_title = "当前图参数下的全部特征组合综合排名 (dG + Trustworthiness + Continuity)"
    else:
        combo_feature_names = top_features
        combo_title = "当前图参数下的候选特征组合综合排名 (基于高分特征 shortlist)"

    combo_df = run_combination_experiment_scientific(
        features,
        combo_feature_names,
        x,
        edge_index,
        n,
        graph_stats['avg_degree'],
        num_features=args.combo_size,
        epochs=args.search_epochs,
    )
    logger.info(print_combination_table(combo_df, combo_title))

    best_combo = tuple(combo_df.iloc[0]['combination'])
    
    best_msg = []
    best_msg.append("\n" + "=" * 90)
    best_msg.append(f"🏆 当前图参数下的最优特征组合: {best_combo}")
    best_msg.append("=" * 90)
    logger.info("\n".join(best_msg))

    if dataset_name.lower() == 'moon' and not args.skip_param_search:
        logger.info("🌙 在 moon 数据集上执行图参数综合评分搜索...")
        param_df = run_parameter_search_moon(
            x,
            D,
            n,
            percentiles=args.search_percentiles,
            scaling_factors=args.search_scales,
            top_feature_count=args.search_top_features,
            num_features=args.combo_size,
            epochs=args.search_epochs,
        )

        if not param_df.empty:
            logger.info(print_parameter_table(param_df, "moon 数据集上的图参数综合评分排名"))
            best_param = param_df.iloc[0]
            
            final_msg = []
            final_msg.append("\n" + "=" * 110)
            final_msg.append(
                "🏁 最终建议: "
                f"percentile={best_param['percentile']:.2f}, "
                f"scale={best_param['scaling_factor']:.2f}, "
                f"best_combination={tuple(best_param['best_combination'])}, "
                f"dG={best_param['dG']:.4f}, "
                f"trust={best_param['trustworthiness']:.4f}, "
                f"continuity={best_param['continuity']:.4f}"
            )
            final_msg.append("=" * 110)
            logger.info("\n".join(final_msg))
        else:
            logger.warning("参数搜索没有得到有效结果")

if __name__ == "__main__":
    main()
