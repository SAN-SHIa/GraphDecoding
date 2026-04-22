import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import networkx as nx

def visualize_results(logger, x, A_eball, viz_results, n, dataset_name="moon", viz_cfg=None):
    logger.info("🎨 visualization...")

    if viz_cfg is None:
        viz_cfg = {
            "keep_percentile": 97.0,
            "fallback_keep_percentile": 95.0,
            "min_keep_ratio": 0.5,
        }

    rec_simple_eball, loss_simple_eball = viz_results.get("GraphSAGE_SimpleScale_EBALL", (None, 0))
    rec_simple_knn, loss_simple_knn = viz_results.get("GraphSAGE_SimpleScale_KNN", (None, 0))

    c = x[:, 0].argsort().argsort()

    center = np.median(x, axis=0)
    dists = np.linalg.norm(x - center, axis=1)
    pct = float(viz_cfg.get("keep_percentile", 97.0))
    thresh = np.percentile(dists, pct)
    keep_mask = dists <= thresh

    min_keep_ratio = float(viz_cfg.get("min_keep_ratio", 0.5))
    if keep_mask.sum() < max(10, int(min_keep_ratio * n)):
        pct = float(viz_cfg.get("fallback_keep_percentile", 95.0))
        thresh = np.percentile(dists, pct)
        keep_mask = dists <= thresh

    kept_xy = x[keep_mask]
    if kept_xy.shape[0] > 0:
        min_xy = kept_xy.min(axis=0)
        max_xy = kept_xy.max(axis=0)
        pad = (max_xy - min_xy) * 0.05 + 1e-6
        xlim = (min_xy[0] - pad[0], max_xy[0] + pad[0])
        ylim = (min_xy[1] - pad[1], max_xy[1] + pad[1])
    else:
        xlim = None
        ylim = None

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig = plt.figure(figsize=(14, 8))

    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(x[keep_mask, 0], x[keep_mask, 1], c=c[keep_mask], s=10, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#eeeeee')
    txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if os.path.exists('./imgs/visible.png'):
        visible = plt.imread('./imgs/visible.png')
        visible_ax = fig.add_axes([0.24, 0.77, 0.1, 0.1], anchor='NE', zorder=1)
        visible_ax.imshow(visible)
        visible_ax.axis('off')

    fr, to = A_eball.nonzero()
    keep_edge_mask = keep_mask[fr] & keep_mask[to]
    fr_f = fr[keep_edge_mask]
    to_f = to[keep_edge_mask]
    G = nx.DiGraph()
    
    # 1. 随机采样 20% 的节点
    np.random.seed(0)
    num_nodes_to_keep = max(1, int(n * 0.20))
    sampled_nodes = set(np.random.choice(n, num_nodes_to_keep, replace=False))
    
    # 2. 从全量边中，只保留两端都在被采样节点集合中的边
    # 不再进行二次抽样，保留子图中 100% 的边，使连线更密集明显
    edges = list(zip(fr, to))
    sampled_edges = [(u, v) for u, v in edges if u in sampled_nodes and v in sampled_nodes]
    
    # 3. 将采样后的节点和边加入图中
    G.add_nodes_from(sampled_nodes)
    G.add_edges_from(sampled_edges)

    ax = fig.add_subplot(2, 2, 2)
    
    # spring_layout: 增加 iterations (迭代次数) 和减少 k (斥力) 让图更乱、更紧凑
    pos = nx.spring_layout(G, k=0.18, iterations=50, seed=0)
    
    # 获取节点的二维坐标并做归一化，让图撑满坐标轴
    pos_ary = np.array(list(pos.values()))
    if pos_ary.shape[0] > 0:
        pos_min = pos_ary.min(axis=0)
        pos_max = pos_ary.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range == 0] = 1  # 防止除以 0
        
        # 将坐标缩放到 [-1, 1] 区间（spring_layout 默认的区间），或者直接缩放到 xlim/ylim 的大小
        for node in pos:
            pos[node] = ((pos[node] - pos_min) / pos_range) * 2 - 1

    # 画图: 线宽(width)调大，边透明度(alpha)调高，让线更明显
    nx.draw_networkx(G, ax=ax, pos=pos, node_size=2, node_color='#005aff', 
                     labels={i: '' for i in G.nodes()}, 
                     edge_color='#84919e', width=0.1, alpha=0.6, arrows=False)

    txt = ax.text(0.05, 0.05, 'Input Graph', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    ax.set_rasterization_zorder(3)
    # Ensure this subplot has axes visible just like the others
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove ax.axis('off') to keep the bounding box
    
    # 强制将这幅图的显示范围锁定在 [-1.1, 1.1] 之间，防止它被缩得太小
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    if rec_simple_eball is not None:
        ax = fig.add_subplot(2, 2, 3)
        ax.scatter(rec_simple_eball[keep_mask, 0], rec_simple_eball[keep_mask, 1], c=c[keep_mask], s=10, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = ax.text(0.05, 0.05, f'SimpleScale (E-ball) $d_G = {loss_simple_eball:.2f}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    if rec_simple_knn is not None:
        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(rec_simple_knn[keep_mask, 0], rec_simple_knn[keep_mask, 1], c=c[keep_mask], s=10, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        txt = ax.text(0.05, 0.05, f'SimpleScale (KNN) $d_G = {loss_simple_knn:.2f}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    fig.subplots_adjust()

    if not os.path.exists('visualize'):
        os.mkdir('visualize')

    save_path = 'visualize/{}_{}.png'.format(datetime.datetime.now().strftime('%m%d%H%M'), dataset_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    logger.info(f"📂 Figure saved: {save_path}")