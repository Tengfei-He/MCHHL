# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import numpy as np


def get_percolated_cliques_from_edges(edge_csv, k, output_csv):
    """
    从输入的边列表计算 CPM 超边，并保存结果为 CSV 文件。

    参数：
        edge_csv (str): 边列表的 CSV 文件路径，包含两列 `id1` 和 `id2`。
        k (int): CPM 算法中的最小 k-派系阈值。
        output_csv (str): 输出的超边格式 CSV 文件路径。

    输出：
        超边格式的 CSV 文件，包含两列：`Node` 和 `Hyperedge`。
    """
    # 1. 从 CSV 文件加载边
    print(1)
    edges = pd.read_csv(edge_csv,header=None)
    edges = edges.sample(n=50000, random_state=42)
    # print(edges)
    edge_list = edges.values.tolist()  # 转为边列表

    print(2)
    # 2. 构建图
    G = nx.Graph()
    G.add_edges_from(edge_list)

    print(3)
    # 3. CPM 算法计算派系
    cliques = list(frozenset(c) for c in nx.find_cliques(G) if len(c) >= k)
    print(cliques)
    matrix = np.zeros((len(cliques), len(cliques)))

    for i in range(len(cliques)):
        for j in range(len(cliques)):
            if i == j:
                n = len(cliques[i])
                if n >= k:
                    matrix[i][j] = 1
            else:
                n = len(cliques[i].intersection(cliques[j]))
                if n >= k - 1:
                    matrix[i][j] = 1

    community_labels = list(range(len(cliques)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1 and i != j:
                community_labels[j] = community_labels[i]

    unique_labels = list(set(community_labels))
    communities = []
    for label in unique_labels:
        community = frozenset.union(*[cliques[j] for j in range(len(community_labels)) if community_labels[j] == label])
        communities.append(community)
    print(4)
    # 4. 转换为超边格式
    hyperedge_format = []
    for hyperedge_id, community in enumerate(communities):
        for node in community:
            hyperedge_format.append((node, hyperedge_id))
    print(5)
    # 5. 保存为 CSV 文件
    df = pd.DataFrame(hyperedge_format, columns=["Node", "Hyperedge"])
    df.to_csv(output_csv, index=False)
    print(f"超边结果已保存到 {output_csv}")


# 示例：调用函数
if __name__ == "__main__":
    # 输入 CSV 文件路径
    edge_csv = "/root/autodl-tmp/MCHNN/Data_mine/drug_drug_index.csv"  # 输入边列表文件路径
    output_csv = "/root/autodl-tmp/MCHNN/Data_mine/source_cpm_drug_hyperedges.csv"  # 输出超边文件路径
    k = 3  # CPM 算法中的最小 k-派系阈值

    # 调用函数计算并保存结果
    get_percolated_cliques_from_edges(edge_csv, k, output_csv)

