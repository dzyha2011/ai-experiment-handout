# K值选择方法实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载和预处理数据
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. 手肘法
def elbow_method(X, max_k=10):
    sse = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    return k_range, sse

# 2. 轮廓系数法
def silhouette_method(X, max_k=10):
    sil_scores = []
    k_range = range(2, max_k + 1)  # 轮廓系数要求K≥2
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
    
    return k_range, sil_scores

# 3. Gap统计量（简化版）
def gap_statistic(X, max_k=10, n_refs=10):
    gaps = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        # 实际数据的SSE
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        actual_sse = kmeans.inertia_
        
        # 随机数据的SSE（多次采样取平均）
        ref_sses = []
        for _ in range(n_refs):
            # 生成与原数据同分布的随机数据
            random_data = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42)
            kmeans_ref.fit(random_data)
            ref_sses.append(kmeans_ref.inertia_)
        
        ref_sse_mean = np.mean(ref_sses)
        gap = np.log(ref_sse_mean) - np.log(actual_sse)
        gaps.append(gap)
    
    return k_range, gaps

# 执行三种方法
k_range_elbow, sse_values = elbow_method(X_scaled)
k_range_sil, sil_scores = silhouette_method(X_scaled)
k_range_gap, gap_values = gap_statistic(X_scaled)

# 可视化结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 手肘法
axes[0].plot(k_range_elbow, sse_values, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('K值')
axes[0].set_ylabel('SSE')
axes[0].set_title('手肘法')
axes[0].grid(True, alpha=0.3)

# 轮廓系数法
axes[1].plot(k_range_sil, sil_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('K值')
axes[1].set_ylabel('平均轮廓系数')
axes[1].set_title('轮廓系数法')
axes[1].grid(True, alpha=0.3)

# Gap统计量
axes[2].plot(k_range_gap, gap_values, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('K值')
axes[2].set_ylabel('Gap统计量')
axes[2].set_title('Gap统计量法')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('k_selection_methods.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出最优K值建议
print("=== K值选择结果 ===")
print(f"轮廓系数法推荐K值: {k_range_sil[np.argmax(sil_scores)]}")
print(f"Gap统计量法推荐K值: {k_range_gap[np.argmax(gap_values)]}")
print("手肘法需要人工观察拐点确定K值")

# 输出详细数值
print("\n=== 详细数值结果 ===")
print("手肘法 SSE值:", [f"{sse:.2f}" for sse in sse_values])
print("轮廓系数值:", [f"{score:.4f}" for score in sil_scores])
print("Gap统计量值:", [f"{gap:.4f}" for gap in gap_values])