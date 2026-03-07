import numpy as np
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_swiss_roll,
    make_s_curve,
    make_blobs
)


def generate_moon(n=5000, noise=0.05):
    """双月形数据集"""
    x, _ = make_moons(n_samples=n, noise=noise, random_state=0)
    return x, "Moon"


def generate_circles(n=5000, noise=0.05, factor=0.5):
    """同心圆环数据集"""
    x, _ = make_circles(n_samples=n, noise=noise, factor=factor, random_state=0)
    return x, "Circles"


def generate_spiral(n=5000, noise=0.5):
    """双螺旋数据集"""
    theta = np.sqrt(np.random.rand(n // 2)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.stack([np.cos(theta) * r_a, np.sin(theta) * r_a], axis=1)

    r_b = -2 * theta - np.pi
    data_b = np.stack([np.cos(theta) * r_b, np.sin(theta) * r_b], axis=1)

    x = np.vstack([data_a, data_b])
    x += np.random.randn(n, 2) * noise
    return x, "Spiral"


def generate_swiss_roll_2d(n=5000, noise=0.5):
    """瑞士卷投影到2D"""
    x_3d, t = make_swiss_roll(n_samples=n, noise=noise, random_state=0)
    x = x_3d[:, [0, 2]]
    return x, "SwissRoll2D"


def generate_s_curve_2d(n=5000, noise=0.1):
    """S曲线投影到2D"""
    x_3d, t = make_s_curve(n_samples=n, noise=noise, random_state=0)
    x = x_3d[:, [0, 2]]
    return x, "SCurve2D"


def generate_clusters(n=5000, n_centers=5, cluster_std=0.5):
    """高斯簇数据集"""
    x, _ = make_blobs(n_samples=n, centers=n_centers, cluster_std=cluster_std, random_state=0)
    return x, "Clusters"


def generate_grid(n=5000, noise=0.1):
    """网格数据集"""
    side = int(np.sqrt(n))
    n = side * side
    xx, yy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    x = np.stack([xx.ravel(), yy.ravel()], axis=1)
    x += np.random.randn(n, 2) * noise
    return x, "Grid"


def generate_ring(n=5000, noise=0.05):
    """单环数据集"""
    theta = np.random.rand(n) * 2 * np.pi
    r = 1.0 + np.random.randn(n) * noise
    x = np.stack([np.cos(theta) * r, np.sin(theta) * r], axis=1)
    return x, "Ring"


def generate_line(n=5000, noise=0.1):
    """线段数据集"""
    t = np.linspace(0, 1, n)
    x = np.stack([t, t + np.random.randn(n) * noise], axis=1)
    return x, "Line"


def generate_wave(n=5000, noise=0.1):
    """波浪数据集"""
    t = np.linspace(0, 4 * np.pi, n)
    x = np.stack([t, np.sin(t) + np.random.randn(n) * noise], axis=1)
    return x, "Wave"


def get_all_datasets(n=5000):
    """获取所有数据集"""
    datasets = [
        generate_moon(n),
        generate_circles(n),
        generate_spiral(n),
        generate_swiss_roll_2d(n),
        generate_s_curve_2d(n),
        generate_clusters(n),
        generate_grid(n),
        generate_ring(n),
        generate_line(n),
        generate_wave(n),
    ]
    return datasets


def get_dataset_by_name(name, n=5000):
    """根据名称获取数据集"""
    dataset_map = {
        "moon": generate_moon,
        "circles": generate_circles,
        "spiral": generate_spiral,
        "swissroll2d": generate_swiss_roll_2d,
        "scurve2d": generate_s_curve_2d,
        "clusters": generate_clusters,
        "grid": generate_grid,
        "ring": generate_ring,
        "line": generate_line,
        "wave": generate_wave,
    }
    func = dataset_map.get(name.lower())
    if func:
        return func(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
