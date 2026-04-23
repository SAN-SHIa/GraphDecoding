# GraphDecoding - 基于图神经网络的隐藏特征恢复

[![arXiv](https://img.shields.io/badge/arXiv-2301.10956-b31b1b.svg)](https://arxiv.org/abs/2301.10956)

毕业设计改进仓库，原始论文：[Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure](https://arxiv.org/abs/2301.10956)  
Ryoma Sato, ICML 2023

---

## 算法思想

**核心问题**：给定一个图结构（仅包含节点和边的连接信息），能否恢复出节点潜在的隐藏几何特征？

**SimpleScale 方法**：
1. **输入**：随机/平凡节点特征 + 结构描述符（PageRank、度、聚类系数等）
2. **架构**：基于 GraphSAGE 的消息传递 + 密度感知缩放（density-aware scaling）
3. **目标**：从纯图结构中重建隐藏几何/流形

![Framework](imgs/framework.PNG)

---

## 快速启动

```bash
# 安装依赖
cd GraphDecoding

uv venv --python 3.12
uv pip install -r requirements.txt

source venvv/bin/activate

# 主实验（moon 等合成数据集）
python src/main.py --dataset moon

# 全量实验
bash run.sh
```

---

## 实验结果

在多种合成数据集上的隐藏维度恢复对比（Eball Avg degree 表示 e-ball 方法达到最优的 K 近邻数）：

| 数据集 | KNN Avg degree | e-ball Avg degree | KNN (dG) | e-ball (dG) | 较优方法 |
|--------|----------------|-------------------|----------|-------------|----------|
| moon | 1005 | 999 | 0.0864 | 0.0051 | e-ball |
| circles | 1090 | 1067 | 0.5997 | 0.0735 | e-ball |
| spiral | 1041 | 1023.8 | 97.2098 | 1.3757 | e-ball |
| swissroll2d | 753 | 755 | 0.5416 | 0.3908 | e-ball |
| scurve2d | 781 | 774 | 0.0351 | 0.0562 | KNN |
| clusters | 882 | 890 | 1.5990 | 6.5253 | KNN |
| grid | 1502 | 1506 | 0.0623 | 0.0964 | e-ball |
| ring | 744 | 743 | 0.0110 | 0.0189 | KNN |
| line | 1195 | 1204 | 0.0878 | 0.0105 | e-ball |
| wave | 699 | 702 | 11.8006 | 0.3658 | e-ball |

---

## 项目结构

```
GraphDecoding/
├── configs/default.yaml      # 配置文件
├── src/
│   ├── main.py               # 主实验入口
│   ├── adult.data            # Adult 数据集
│   └── utils/                # 工具模块
│       ├── model.py          # SimpleScale 模型
│       ├── datasets.py       # 数据集生成
│       ├── visualization.py  # 可视化
│       └── logging.py        # 日志
├── tools/
│   ├── semi_adult.py         # Adult 数据集实验
│   └── feature_analysis.py   # 特征分析
├── outputs/                   # 实验输出
├── imgs/                      # 算法框架图
├── requirements.txt
└── README.md
```

---

## 引用

```bibtex
@inproceedings{sato2023graph,
  author    = {Ryoma Sato},
  title     = {Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure},
  booktitle = {International Conference on Machine Learning, {ICML}},
  year      = {2023},
}
```
