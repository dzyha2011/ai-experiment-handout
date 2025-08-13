# 聚类结果评估完整代码
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_samples, confusion_matrix
)
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载和预处理数据
iris = load_iris()
X = iris.data
y_true = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# 1. 外部评估指标
print("=== 外部评估指标（需要真实标签） ===")
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)

print(f"调整兰德指数 (ARI): {ari:.4f}")
print(f"标准化互信息 (NMI): {nmi:.4f}")
print(f"V-measure: {v_measure:.4f}")

# 2. 内部评估指标
print("\n=== 内部评估指标（无需真实标签） ===")
sil_score = silhouette_score(X_scaled, y_pred)
ch_score = calinski_harabasz_score(X_scaled, y_pred)
db_score = davies_bouldin_score(X_scaled, y_pred)

print(f"平均轮廓系数: {sil_score:.4f}")
print(f"Calinski-Harabasz指数: {ch_score:.2f}")
print(f"Davies-Bouldin指数: {db_score:.4f}")
print(f"簇内平方和 (SSE): {kmeans.inertia_:.2f}")

# 3. 轮廓系数详细分析
sample_silhouette_values = silhouette_samples(X_scaled, y_pred)

# 可视化轮廓系数分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 轮廓系数分布图
y_lower = 10
for i in range(3):
    cluster_silhouette_values = sample_silhouette_values[y_pred == i]
    cluster_silhouette_values.sort()
    
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.nipy_spectral(float(i) / 3)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_xlabel('轮廓系数值')
ax1.set_ylabel('簇标签')
ax1.set_title('各簇轮廓系数分布')
ax1.axvline(x=sil_score, color="red", linestyle="--", label=f'平均值: {sil_score:.3f}')
ax1.legend()

# 混淆矩阵热力图
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')
ax2.set_title('聚类结果混淆矩阵')

plt.tight_layout()
plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 不同K值的评估对比
k_range = range(2, 8)
metrics = {
    'silhouette': [],
    'calinski_harabasz': [],
    'davies_bouldin': [],
    'sse': []
}

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    
    metrics['silhouette'].append(silhouette_score(X_scaled, labels_temp))
    metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels_temp))
    metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels_temp))
    metrics['sse'].append(kmeans_temp.inertia_)

# 可视化不同K值的评估指标
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(k_range, metrics['silhouette'], 'bo-', linewidth=2)
axes[0,0].set_title('轮廓系数 vs K值')
axes[0,0].set_xlabel('K值')
axes[0,0].set_ylabel('轮廓系数')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(k_range, metrics['calinski_harabasz'], 'ro-', linewidth=2)
axes[0,1].set_title('Calinski-Harabasz指数 vs K值')
axes[0,1].set_xlabel('K值')
axes[0,1].set_ylabel('CH指数')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(k_range, metrics['davies_bouldin'], 'go-', linewidth=2)
axes[1,0].set_title('Davies-Bouldin指数 vs K值')
axes[1,0].set_xlabel('K值')
axes[1,0].set_ylabel('DB指数')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(k_range, metrics['sse'], 'mo-', linewidth=2)
axes[1,1].set_title('SSE vs K值')
axes[1,1].set_xlabel('K值')
axes[1,1].set_ylabel('SSE')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n评估分析完成！结果图片已保存。")

# 输出详细的评估报告
print("\n=== 详细评估报告 ===")
print(f"数据集: Iris鸢尾花数据集 (150个样本, 4个特征)")
print(f"聚类算法: K-means (K=3)")
print(f"数据预处理: 标准化")
print("\n外部评估指标:")
print(f"  - ARI (调整兰德指数): {ari:.4f} (范围[-1,1], 越接近1越好)")
print(f"  - NMI (标准化互信息): {nmi:.4f} (范围[0,1], 越接近1越好)")
print(f"  - V-measure: {v_measure:.4f} (范围[0,1], 越接近1越好)")
print("\n内部评估指标:")
print(f"  - 轮廓系数: {sil_score:.4f} (范围[-1,1], 越接近1越好)")
print(f"  - CH指数: {ch_score:.2f} (无固定范围, 越大越好)")
print(f"  - DB指数: {db_score:.4f} (最小值0, 越小越好)")
print(f"  - SSE: {kmeans.inertia_:.2f} (越小表示簇内越紧凑)")

# 聚类质量评价
if ari > 0.7:
    quality = "优秀"
elif ari > 0.5:
    quality = "良好"
elif ari > 0.3:
    quality = "一般"
else:
    quality = "较差"
    
print(f"\n综合评价: 聚类质量{quality} (ARI={ari:.4f})")
print("建议: K-means算法在Iris数据集上表现良好，成功识别了3个鸢尾花类别的主要特征。")