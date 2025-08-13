# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征数据 (150, 4)
y_true = iris.target  # 真实标签 (150,)
feature_names = iris.feature_names

print(f"数据集形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别标签: {np.unique(y_true)}")

# 2. 数据预处理（标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-means模型训练
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# 4. 聚类结果可视化（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制聚类结果散点图
plt.figure(figsize=(12, 5))

# 子图1：聚类结果
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', 
                      alpha=0.7, edgecolors='k')
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, 
            c='red', linewidths=3, label='质心')
plt.xlabel('PCA特征1')
plt.ylabel('PCA特征2')
plt.title('K-means聚类结果')
plt.legend()
plt.colorbar(scatter, label='簇标签')

# 子图2：真实标签
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', 
                       alpha=0.7, edgecolors='k')
plt.xlabel('PCA特征1')
plt.ylabel('PCA特征2')
plt.title('真实标签分布')
plt.colorbar(scatter2, label='真实类别')

plt.tight_layout()
plt.savefig('iris_clustering_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 聚类评估
ari = adjusted_rand_score(y_true, y_pred)
sil_score = silhouette_score(X_scaled, y_pred)

print(f"\n=== 聚类评估结果 ===")
print(f"调整兰德指数 (ARI): {ari:.4f}")
print(f"平均轮廓系数: {sil_score:.4f}")
print(f"簇内平方和 (SSE): {kmeans.inertia_:.2f}")

# 6. K值选择实验（手肘法）
sse_values = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X_scaled)
    sse_values.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K值')
plt.ylabel('SSE (簇内平方和)')
plt.title('手肘法确定最优K值')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n实验完成！结果图片已保存。")