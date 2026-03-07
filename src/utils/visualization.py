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
    G.add_edges_from(zip(fr_f, to_f))

    ax = fig.add_subplot(2, 2, 2)
    pos = {i: x[i] for i in range(n)}
    kept_nodes = [int(i) for i in range(n) if keep_mask[i]]
    pos_kept = {i: pos[i] for i in kept_nodes}
    nx.draw_networkx_nodes(G, pos_kept, ax=ax, node_size=0.5, node_color='#005aff')

    if G.number_of_edges() > 2000:
        edges_to_draw = list(G.edges())[:2000]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_to_draw, edge_color='#84919e', width=0.0005, arrowsize=0.1)
    else:
        nx.draw_networkx_edges(G, pos_kept, ax=ax, edge_color='#84919e', width=0.0005, arrowsize=0.1)

    txt = ax.text(0.05, 0.05, 'Input Graph (Real Pos)', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    ax.set_rasterization_zorder(3)
    ax.axis('off')
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

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
