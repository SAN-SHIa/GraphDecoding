"""
分析比较不同的流形学图特征参数，选出最有价值的特征组合用于神经网络
"""

import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns

from util import GraphSAGE_SimpleScale, moon, stationary, dG, seed_everything, setup_logger
from datasets_2d import get_dataset_by_name
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
    
    edge_index = np.vstack([fr, to]) if len(fr) > 0 else np.empty((2, 0))
    return A, edge_index


def compute_all_features(A, D, n):
    """
    计算所有候选特征
    返回特征字典：{特征名: 特征向量}
    """
    features = {}
    
    # 1. PageRank (稳态分布)
    # 物理含义: 全局重要性，流形稠密区域通常PR值高
    # 公式: PR(i) = alpha * sum(PR(j)/L(j)) + (1-alpha)/N
    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    features['pagerank'] = pr / (np.mean(pr) + 1e-9)
    
    # 2. 度数相关特征
    # 物理含义: 局部连接密度，最基础的拓扑特征
    # 公式: d_i = sum(A_ij)
    A_bool = (A > 0)
    degree = np.array(A_bool.sum(axis=1)).flatten()
    features['degree'] = degree / (np.mean(degree) + 1e-9)
    
    # 3. 入度
    # 物理含义: 汇聚程度，有多少点把该点视为邻居
    in_degree = np.array(A_bool.sum(axis=0)).flatten()
    features['in_degree'] = in_degree / (np.mean(in_degree) + 1e-9)
    
    # 4. 出度
    # 物理含义: 局部几何密度(对于e-ball图)，半径内的邻居数
    out_degree = np.array(A_bool.sum(axis=1)).flatten()
    features['out_degree'] = out_degree / (np.mean(out_degree) + 1e-9)
    
    # 5. 聚类系数
    # 物理含义: 局部团块结构，流形平坦程度(邻居的邻居也是邻居)
    # 公式: C_i = 2 * e_i / (d_i * (d_i - 1))
    A_dense = A_bool.astype(np.float32).toarray()
    A2 = A_dense @ A_dense
    diag_A3 = np.einsum('ij,ij->i', A2, A_dense)
    denom = degree * (degree - 1)
    clust = np.zeros_like(degree, dtype=np.float32)
    mask = denom > 0
    clust[mask] = diag_A3[mask] / denom[mask]
    features['clustering'] = clust / (np.mean(clust) + 1e-9)
    
    # 6. 局部密度估计 (基于邻居距离)
    # 物理含义: 物理空间密度，距离逆相关
    # 公式: rho ~ 1 / mean(dist)
    k = min(10, n - 1)
    sorted_D = np.sort(D, axis=1)
    local_density = 1.0 / (np.mean(sorted_D[:, 1:k+1], axis=1) + 1e-9)
    features['local_density'] = local_density / (np.mean(local_density) + 1e-9)
    
    # 7. k近邻距离的标准差 (局部变异性)
    # 物理含义: 局部几何均匀性，大方差意味着处于边界或噪声
    # 公式: std(dist_1...dist_k)
    knn_std = np.std(sorted_D[:, 1:k+1], axis=1)
    features['knn_variance'] = knn_std / (np.mean(knn_std) + 1e-9)
    
    # 8. 二阶邻居数 (2-hop)
    # 物理含义: 扩展范围的局部密度
    # 公式: sum(A^2 > 0)
    A2_bool = (A2 > 0)
    two_hop = np.array(A2_bool.sum(axis=1)).flatten()
    features['two_hop_neighbors'] = two_hop / (np.mean(two_hop) + 1e-9)
    
    # 9. 邻居的平均度数
    # 物理含义: 邻域一致性，区分核心与边缘挂件
    # 公式: mean(degree(neighbors))
    neighbor_deg_sum = A_bool @ degree
    avg_neighbor_deg = np.zeros(n)
    mask = degree > 0
    avg_neighbor_deg[mask] = neighbor_deg_sum[mask] / degree[mask]
    features['avg_neighbor_degree'] = avg_neighbor_deg / (np.mean(avg_neighbor_deg) + 1e-9)
    
    # 10. 核心数近似 (基于迭代度数修剪)
    # 物理含义: 节点在网络中的深度/层次，k-core子图层级
    # 计算: 递归剥离度数<k的节点
    G_nx = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    try:
        core_number = nx.core_number(G_nx)
        core = np.array([core_number.get(i, 0) for i in range(n)])
        features['core_number'] = core / (np.mean(core) + 1e-9)
    except:
        features['core_number'] = np.ones(n)
    
    # 11. 介数中心性近似 (采样加速)
    # 物理含义: 桥梁作用，位于流形瓶颈处的点值高
    # 公式: sum(sigma_st(i) / sigma_st)
    try:
        if n > 500:
            betweenness = nx.betweenness_centrality(G_nx, k=min(100, n))
        else:
            betweenness = nx.betweenness_centrality(G_nx)
        bet = np.array([betweenness.get(i, 0) for i in range(n)])
        features['betweenness'] = bet / (np.mean(bet) + 1e-9)
    except:
        features['betweenness'] = np.ones(n)
    
    # 12. 特征向量中心性
    # 物理含义: 邻居重要性递归，反映全局几何低频信息
    # 公式: A x = lambda x
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_nx)
        eig = np.array([eigenvector.get(i, 0) for i in range(n)])
        features['eigenvector'] = eig / (np.mean(eig) + 1e-9)
    except:
        features['eigenvector'] = np.ones(n)
    
    # 13. 接近中心性 (采样)
    # 物理含义: 几何中心程度，到其他所有点的平均距离倒数
    # 公式: (N-1) / sum(dist(i,j))
    try:
        closeness = nx.closeness_centrality(G_nx)
        close = np.array([closeness.get(i, 0) for i in range(n)])
        features['closeness'] = close / (np.mean(close) + 1e-9)
    except:
        features['closeness'] = np.ones(n)
    
    # 14. 边权重和
    # 物理含义: 加权局部密度 (权重=1/dist)
    # 公式: sum(W_ij)
    weight_sum = np.array(A.sum(axis=1)).flatten()
    features['weight_sum'] = weight_sum / (np.mean(weight_sum) + 1e-9)
    
    # 处理 NaN 和 Inf
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
    
    for name, feat in features.items():
        feat = feat.reshape(-1)
        
        # 与 x 坐标的相关性
        pearson_x, _ = pearsonr(feat, x[:, 0])
        spearman_x, _ = spearmanr(feat, x[:, 0])
        
        # 与 y 坐标的相关性
        pearson_y, _ = pearsonr(feat, x[:, 1])
        spearman_y, _ = spearmanr(feat, x[:, 1])
        
        # 互信息
        mi_x = mutual_info_regression(feat.reshape(-1, 1), x[:, 0], random_state=0)[0]
        mi_y = mutual_info_regression(feat.reshape(-1, 1), x[:, 1], random_state=0)[0]
        
        # 综合得分 (取最大绝对值)
        max_pearson = max(abs(pearson_x), abs(pearson_y))
        max_spearman = max(abs(spearman_x), abs(spearman_y))
        max_mi = max(mi_x, mi_y)
        
        results[name] = {
            'pearson_x': pearson_x,
            'pearson_y': pearson_y,
            'spearman_x': spearman_x,
            'spearman_y': spearman_y,
            'mi_x': mi_x,
            'mi_y': mi_y,
            'max_pearson': max_pearson,
            'max_spearman': max_spearman,
            'max_mi': max_mi,
            'combined_score': (max_pearson + max_spearman + max_mi) / 3
        }
    
    return results


def train_with_features(x, edge_index, feature_matrix, n, m, epochs=50):
    """使用给定特征矩阵训练模型并返回 dG 分数"""
    seed_everything(0)
    
    K = int(np.sqrt(n) * np.log2(n) / 10)
    X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)
    eye_n = torch.eye(n)
    x_tensor = torch.FloatTensor(x)
    density = torch.FloatTensor(feature_matrix)
    
    n_train = int(n * 0.7)
    train_ind = torch.randperm(n)[:n_train]
    
    net = GraphSAGE_SimpleScale(m)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    net.train()
    
    for _ in range(epochs):
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
    
    # 最终评估
    with torch.no_grad():
        idx = torch.randperm(n)[:m]
        ind = eye_n[:, idx]
        X_extended = torch.hstack([X, ind])
        X_with_density = torch.cat([X_extended, density], dim=1)
        data = Data(x=X_with_density, edge_index=edge_index)
        rec = net(data)
        rec_np = rec.numpy()
        R, _ = orthogonal_procrustes(x, rec_np)
        aligned = rec_np @ R.T
        score = float(dG(x_tensor, torch.FloatTensor(aligned)))
    
    return score


def run_combination_experiment(features, feature_names, x, edge_index, n, m, num_features=3):
    """测试所有 C(n,3) 特征组合"""
    results = []
    
    all_combinations = list(combinations(feature_names, num_features))
    logger.info(f"📊 测试 {len(all_combinations)} 种特征组合...")
    
    for combo in tqdm.tqdm(all_combinations, desc="Testing combinations"):
        # 构建特征矩阵
        feat_list = [features[name].reshape(-1, 1) for name in combo]
        feature_matrix = np.hstack(feat_list)
        
        # 标准化
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
        
        # 训练并评估
        score = train_with_features(x, edge_index, feature_matrix, n, m, epochs=30)
        results.append((combo, score))
    
    # 按分数排序 (越低越好)
    results.sort(key=lambda x: x[1])
    
    return results


def visualize_feature_correlation(features, dataset_name="moon"):
    """
    可视化不同特征之间的相关性热力图
    
    Parameters:
    -----------
    features : dict
        特征字典 {特征名: 特征向量}
    dataset_name : str
        数据集名称，用于保存文件
    """
    feature_names = list(features.keys())
    n_features = len(feature_names)
    
    # 构建特征矩阵
    feature_matrix = np.column_stack([features[name] for name in feature_names])
    
    # 计算 Pearson 相关系数矩阵
    pearson_corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            pearson_corr[i, j], _ = pearsonr(feature_matrix[:, i], feature_matrix[:, j])
    
    # 计算 Spearman 相关系数矩阵
    spearman_corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            spearman_corr[i, j], _ = spearmanr(feature_matrix[:, i], feature_matrix[:, j])
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pearson 相关性热力图
    ax1 = axes[0]
    sns.heatmap(pearson_corr, 
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
                annot_kws={'size': 8})
    ax1.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9)
    
    # Spearman 相关性热力图
    ax2 = axes[1]
    sns.heatmap(spearman_corr, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                square=True,
                ax=ax2,
                annot_kws={'size': 8})
    ax2.set_title('Spearman Correlation', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    if not os.path.exists('visualize'):
        os.mkdir('visualize')
    
    save_path = f'visualize/feature_correlation_{dataset_name}_{datetime.datetime.now().strftime("%m%d%H%M")}.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"📊 特征相关性热力图已保存: {save_path}")
    
    plt.show()
    
    # 打印高度相关的特征对 (|r| > 0.8)
    print("\n" + "="*60)
    print("高度相关的特征对 (|Pearson| > 0.8):")
    print("="*60)
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(pearson_corr[i, j]) > 0.8:
                print(f"  {feature_names[i]} <-> {feature_names[j]}: r = {pearson_corr[i, j]:.3f}")
    
    return pearson_corr, spearman_corr


def main():
    parser = argparse.ArgumentParser(description="Feature Analysis for Graph Decoding")
    parser.add_argument("--dataset", type=str, default="moon", help="Dataset name")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples (smaller for faster analysis)")
    parser.add_argument("--m", type=int, default=100, help="Landmark size")
    parser.add_argument("--full_search", action="store_true", help="Run full combination search")
    args = parser.parse_args()
    
    n = args.n
    m = args.m
    dataset_name = args.dataset
    
    logger.info(f"🚀 特征分析: 数据集={dataset_name}, n={n}")
    
    # 准备数据
    if dataset_name.lower() == "moon":
        from util import moon
        x, n = moon(n)
    else:
        x, _ = get_dataset_by_name(dataset_name, n)
        n = len(x)
    
    D = pairwise_distances(x)
    
    # 构建 ε-ball 图
    base_epsilon = np.percentile(D[D > 0], 5)
    epsilon = base_epsilon * 2.7
    A, edge_index = build_eball_graph(D, epsilon)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    logger.info(f"📈 图构建完成: 节点={n}, 边={A.nnz}")
    
    # 计算所有特征
    logger.info("🔬 计算所有候选特征...")
    features = compute_all_features(A, D, n)
    feature_names = list(features.keys())
    logger.info(f"   共 {len(feature_names)} 个特征: {feature_names}")
    
    # 可视化特征之间的相关性
    logger.info("🎨 生成特征相关性热力图...")
    visualize_feature_correlation(features, dataset_name)
    
    # 评估特征重要性
    logger.info("📊 评估特征重要性...")
    importance = evaluate_feature_importance(features, x)
    
    # 打印特征重要性排名
    print("\n" + "="*80)
    print("特征重要性排名 (综合得分 = Pearson + Spearman + 互信息)")
    print("="*80)
    
    sorted_features = sorted(importance.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    
    print(f"{'排名':<4} {'特征名':<22} {'Pearson':<10} {'Spearman':<10} {'互信息':<10} {'综合得分':<10}")
    print("-"*80)
    
    for rank, (name, scores) in enumerate(sorted_features, 1):
        print(f"{rank:<4} {name:<22} {scores['max_pearson']:<10.4f} {scores['max_spearman']:<10.4f} {scores['max_mi']:<10.4f} {scores['combined_score']:<10.4f}")
    
    # 推荐的前3个特征
    top3 = [name for name, _ in sorted_features[:3]]
    print("\n" + "="*80)
    print(f"📌 基于相关性分析推荐的前3个特征: {top3}")
    print("="*80)
    
    if args.full_search:
        # 运行完整的组合实验
        logger.info("🔥 运行完整组合搜索...")
        combo_results = run_combination_experiment(features, feature_names, x, edge_index, n, m)
        
        print("\n" + "="*80)
        print("特征组合性能排名 (dG 分数越低越好)")
        print("="*80)
        
        for rank, (combo, score) in enumerate(combo_results[:10], 1):
            print(f"{rank:<4} {str(combo):<60} dG={score:.4f}")
        
        best_combo = combo_results[0][0]
        print("\n" + "="*80)
        print(f"🏆 基于训练实验的最优特征组合: {best_combo}")
        print("="*80)
    else:
        # 快速测试: 比较推荐组合 vs 原始组合
        print("\n🔬 快速对比测试...")
        
        # 原始组合: pagerank, in_degree, clustering
        original_combo = ['pagerank', 'in_degree', 'clustering']
        original_matrix = np.vstack([features[name] for name in original_combo]).T
        original_matrix = StandardScaler().fit_transform(original_matrix)
        original_score = train_with_features(x, edge_index, original_matrix, n, m)
        
        # 推荐组合
        recommended_matrix = np.vstack([features[name] for name in top3]).T
        recommended_matrix = StandardScaler().fit_transform(recommended_matrix)
        recommended_score = train_with_features(x, edge_index, recommended_matrix, n, m)
        
        print(f"\n原始组合 {original_combo}: dG = {original_score:.4f}")
        print(f"推荐组合 {top3}: dG = {recommended_score:.4f}")
        
        if recommended_score < original_score:
            print(f"✅ 推荐组合优于原始组合! 改进: {(original_score - recommended_score):.4f}")
        else:
            print(f"ℹ️  原始组合已经是较优选择")


if __name__ == "__main__":
    main()
