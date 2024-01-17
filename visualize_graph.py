import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.cm import ScalarMappable

# 加载模型
model = torch.load('\exps\2024-01-16-13-40-31\final_model.pth')

# 查看模型字典的键
# print(model.keys())
weights_layer1 = model['backbone.gc1.lin.weight'].cpu().data.numpy()
weights_layer2 = model['backbone.gc2.lin.weight'].cpu().data.numpy()

# # 创建子图
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # 绘制第一层权重矩阵的热图
# sns.heatmap(weights_layer1, ax=axes[0], cmap='viridis', annot=False)
# axes[0].set_title('Layer 1 Weights')

# # 绘制第二层权重矩阵的热图
# sns.heatmap(weights_layer2, ax=axes[1], cmap='viridis', annot=False)
# axes[1].set_title('Layer 2 Weights')

# plt.show()






# G = nx.Graph()

# # 添加节点
# num_nodes = weights_layer1.shape[0]
# G.add_nodes_from(range(num_nodes))

# # 添加边及其权重
# for i in range(num_nodes):
#     for j in range(num_nodes):
#         weight = weights_layer1[i, j]
#         G.add_edge(i, j, weight=weight)

# # 绘制图
# edge_weights = [G[i][j]['weight'] for i, j in G.edges()]

# plt.figure(figsize=(10, 10))
# pos = nx.spring_layout(G)  # 使用spring布局算法
# nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500, edge_color=edge_weights, width=edge_weights, cmap=plt.cm.Blues)
# plt.show()


# 创建图
G = nx.Graph()

# 添加节点
num_nodes = weights_layer1.shape[0]
G.add_nodes_from(range(num_nodes))

# 添加边及其权重
for i in range(num_nodes):
    for j in range(num_nodes):
        weight = weights_layer2[i, j]
        G.add_edge(i, j, weight=weight)

# 获取最大的几个权重对应的边
num_top_weights = 5  # 要获取的最大权重的数量
sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:num_top_weights]

# 输出最大权重对应的节点
for edge in sorted_edges:
    print(f"Nodes: {edge[0]}, {edge[1]}, Weight: {edge[2]['weight']}")

# 设置颜色映射
edge_weights = [edge[2]['weight'] for edge in G.edges(data=True)]
edge_colors = plt.cm.viridis(edge_weights)
cmap = plt.cm.viridis

# 绘制图
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # 使用spring布局算法，指定种子以保持一致性
nx.draw(G, pos, with_labels=True, font_size=8, font_weight='bold', node_size=300, edge_color=edge_colors, width=2.0, edge_cmap=cmap)

# 创建可映射对象并添加颜色条
sm = ScalarMappable(cmap=cmap)
sm.set_array(edge_weights)
cbar = plt.colorbar(sm, label='Edge Weight')

plt.title('Graph Visualization of GCN Weights', fontsize=16)
plt.show()