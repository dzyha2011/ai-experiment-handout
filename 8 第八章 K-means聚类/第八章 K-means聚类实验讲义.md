# K-means聚类实验讲义


## 一、聚类分析与 K-means 算法原理

### 1.1 聚类分析基础

聚类分析（Clustering Analysis）是**无监督学习**的核心技术之一，其目标是将数据集按照样本间的相似性划分为若干个互不重叠的子集（称为“簇”，Cluster），使得同一簇内样本的相似度尽可能高，不同簇间样本的相似度尽可能低。与监督学习不同，聚类分析不需要预先标注的训练样本，而是直接从数据本身挖掘内在结构。

**典型应用场景**包括：
- 客户分群（基于消费行为划分客户群体）
- 异常检测（识别远离主要簇的离群点）
- 图像分割（将像素按颜色/纹理相似度分组）
- 基因表达分析（将具有相似表达模式的基因聚类）
- 文档主题发现（将内容相似的文档自动归类）


### 1.2 K-means 算法原理

K-means 是最经典的基于划分的聚类算法，由 MacQueen 在 1967 年提出。其核心思想是通过迭代优化实现“簇内紧凑、簇间分离”的聚类目标。

#### 1.2.1 目标函数
K-means 以**平方误差和（Sum of Squared Errors, SSE）** 作为优化目标，定义为所有样本到其所属簇质心的欧氏距离平方和：

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

其中：
- $K$ 为预设的簇数
- $C_k$ 表示第 $k$ 个簇
- $x_i$ 为簇 $C_k$ 中的样本
- $\mu_k$ 为簇 $C_k$ 的质心（该簇所有样本的均值向量）

算法的目标是找到使 $J$ 最小化的簇划分方案。


#### 1.2.2 算法流程
K-means 算法通过以下步骤迭代优化目标函数：

1. **初始化**：随机选择 $K$ 个样本作为初始质心 $\{\mu_1, \mu_2, ..., \mu_K\}$  
2. **分配样本**：对每个样本 $x_i$，计算其到所有质心的欧氏距离，将其分配到距离最近的质心所属的簇：  
   $$C_k = \{x_i \mid \arg\min_j \|x_i - \mu_j\|^2\}$$  
3. **更新质心**：重新计算每个簇的质心，即簇内所有样本的均值：  
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$  
4. **收敛判断**：若质心位置不再变化（或变化小于阈值），或达到最大迭代次数，则停止；否则返回步骤 2。


#### 1.2.3 算法特性分析

**优点**：
- 原理简单直观，计算效率高（时间复杂度为 $O(nKT)$，其中 $n$ 为样本数，$T$ 为迭代次数）
- 对大规模数据集具有较好的可扩展性
- 聚类结果可解释性强

**缺点**：
- 需要预先指定簇数 $K$，对 $K$ 的选择敏感
- 对初始质心的选择敏感，可能收敛到局部最优解
- 只能发现凸形分布的簇，对非凸形状数据效果差
- 对噪声和离群点敏感（离群点会显著影响质心计算）
- 不适合处理类别不平衡的数据


## 二、K值的选择方法

K-means 算法的性能严重依赖于簇数 $K$ 的选择。若 $K$ 过小，簇内样本差异过大；若 $K$ 过大，会导致簇的过度细分。以下介绍三种常用的 $K$ 值选择方法：


### 2.1 手肘法（Elbow Method）

#### 原理
手肘法基于“随着 $K$ 增大，SSE 逐渐减小”的特性：当 $K$ 小于真实簇数时，SSE 下降迅速；当 $K$ 达到或超过真实簇数后，SSE 下降趋势变缓，形成“手肘”形状的拐点，该拐点对应的 $K$ 即为最优值。

#### 实现步骤
1. 遍历 $K=1,2,...,K_{\text{max}}$（通常取 1~10），对每个 $K$ 训练 K-means 模型  
2. 记录对应的 SSE 值  
3. 绘制 $K$-SSE 曲线，寻找拐点对应的 $K$

#### 代码示例
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris().data
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)  # inertia_属性存储SSE值

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, 'bo-')
plt.xlabel('K值')
plt.ylabel('SSE')
plt.title('手肘法确定最优K值')
plt.show()
```

#### 局限性
- 当数据分布无明显簇结构时，曲线可能无明显拐点
- 主观性较强，不同人可能选择不同拐点


### 2.2 轮廓系数法（Silhouette Coefficient）

#### 原理
轮廓系数综合考虑**类内相似度**和**类间相似度**，定义为：

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中：
- $a(i)$：样本 $i$ 与同一簇内其他样本的平均距离（类内不相似度）
- $b(i)$：样本 $i$ 与最近异簇所有样本的平均距离（类间不相似度）

轮廓系数取值范围为 $[-1, 1]$：
- $s(i) \approx 1$：样本聚类合理
- $s(i) \approx 0$：样本处于两个簇的边界
- $s(i) < 0$：样本可能被错误聚类

**最优 $K$** 对应所有样本轮廓系数的平均值最大。

#### 代码示例
```python
from sklearn.metrics import silhouette_score

sil_scores = []
k_range = range(2, 11)  # 轮廓系数要求K≥2
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)
    sil_scores.append(silhouette_score(data, labels))

plt.figure(figsize=(8, 4))
plt.plot(k_range, sil_scores, 'bo-')
plt.xlabel('K值')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数法确定最优K值')
plt.show()
```

#### 优势与局限
- **优势**：无需真实标签，适用于无监督场景
- **局限**：对凸形分布数据效果好，对非凸分布数据可能失效


### 2.3 Gap统计量（Gap Statistic）

#### 原理
Gap统计量通过比较**实际数据的聚类效果**与**随机数据的聚类效果**来确定 $K$：

$$\text{Gap}(K) = E[\log(SSE_{\text{random}})] - \log(SSE_{\text{actual}})$$

其中 $E[\log(SSE_{\text{random}})]$ 是随机数据 SSE 对数的期望（通过多次随机采样估计）。**最优 $K$** 是使 $\text{Gap}(K)$ 最大且满足 $\text{Gap}(K) \geq \text{Gap}(K+1) - s_{K+1}$（$s_{K+1}$ 为标准差）的最小 $K$。

#### 实现思路
1. 生成与原数据同分布的随机参考数据集（通常通过将各特征范围标准化后均匀采样）
2. 对实际数据和随机数据分别计算不同 $K$ 下的 $\log(SSE)$
3. 计算 Gap 值及标准差，确定最优 $K$

#### 优势
- 理论依据更充分，对各类数据分布适应性更强
- 可量化判断 $K$ 的显著性


## 三、Python 实现 K-means 聚类

### 3.1 实验目标
通过 scikit-learn 内置的 Iris 数据集实现 K-means 聚类，掌握数据预处理、模型训练、结果可视化的完整流程，并通过代码填空加深对关键步骤的理解。


### 3.2 数据集介绍
Iris 数据集包含 3 类鸢尾花（Setosa、Versicolor、Virginica）的 150 个样本，每个样本有 4 个特征：萼片长度、萼片宽度、花瓣长度、花瓣宽度（单位：cm）。


### 3.3 完整实验代码（含代码填空）

```python
# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征数据 (150, 4)
y_true = iris.target  # 真实标签 (150,)
feature_names = iris.feature_names

# 2. 数据探索性分析
print(f"数据集形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别标签: {np.unique(y_true)}")  # 0, 1, 2（对应3类鸢尾花）

# 3. 数据预处理（标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化特征（均值=0，方差=1）

# 4. K-means模型训练（代码填空1：设置簇数K）
kmeans = KMeans(n_clusters=___①___, random_state=42, n_init=10)  # n_init确保多次初始化取最优
y_pred = kmeans.fit_predict(X_scaled)

# 5. 聚类结果可视化（高维数据降维）
# 代码填空2：设置PCA降维后的维度
pca = PCA(n_components=___②___)
X_pca = pca.fit_transform(X_scaled)

# 绘制聚类结果散点图
plt.figure(figsize=(10, 6))
# 代码填空3：设置颜色映射（如'viridis'、'rainbow'等）
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap=___③___, 
                      alpha=0.7, edgecolors='k')
# 绘制质心
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, 
            c='red', linewidths=3, label='质心')
plt.xlabel('PCA特征1')
plt.ylabel('PCA特征2')
plt.title('K-means聚类结果（PCA降维可视化）')
plt.legend()
plt.colorbar(scatter, label='簇标签')
plt.show()
```


### 3.4 关键步骤解释

1. **数据标准化**：K-means 基于欧氏距离计算，标准化可消除特征量纲差异（如花瓣长度和宽度的单位相同但数值范围不同），避免某一特征主导聚类结果。

2. **PCA降维**：Iris 数据集为 4 维特征，通过 PCA 将其降至 2 维，以便可视化。PCA 保留数据中方差最大的方向，尽可能保留原始信息。

3. **模型参数**：`n_init=10` 表示进行 10 次初始质心随机初始化，选择 SSE 最小的模型，缓解初始质心敏感问题。


## 四、K-means 聚类结果评估

聚类评估分为**外部评估**（需真实标签）和**内部评估**（无需真实标签）两类，以下介绍常用指标及实现。


### 4.1 外部评估指标（有真实标签）

#### 4.1.1 调整兰德指数（Adjusted Rand Index, ARI）
ARI 衡量预测簇标签与真实标签的一致性，取值范围为 $[-1, 1]$：
- $ARI = 1$：完全一致
- $ARI = 0$：与随机分配一致
- $ARI < 0$：结果差于随机分配

**代码实现**：
```python
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(y_true, y_pred)
print(f"调整兰德指数 (ARI): {ari:.4f}")
```


#### 4.1.2 标准化互信息（Normalized Mutual Information, NMI）
NMI 基于信息论，衡量两个标签分布的互信息与边缘熵的标准化值，取值范围为 $[0, 1]$：
- $NMI = 1$：完全一致
- $NMI = 0$：无相关性

**代码实现**：
```python
from sklearn.metrics import normalized_mutual_info_score
nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"标准化互信息 (NMI): {nmi:.4f}")
```


### 4.2 内部评估指标（无真实标签）

#### 4.2.1 轮廓系数（Silhouette Coefficient）
如 2.2 节所述，轮廓系数综合类内与类间距离，取值范围 $[-1, 1]$，越大表示聚类效果越好。

**代码实现**：
```python
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_scaled, y_pred)
print(f"平均轮廓系数: {sil_score:.4f}")
```


#### 4.2.2 Calinski-Harabasz 指数
该指数定义为**组间方差与组内方差的比值**，值越大表示簇分离越好：

$$CH = \frac{\text{组间离散度}}{\text{组内离散度}} = \frac{\text{SSB}/(K-1)}{\text{SSW}/(n-K)}$$

其中 $\text{SSB}$ 为组间平方和，$\text{SSW}$ 为组内平方和。

**代码实现**：
```python
from sklearn.metrics import calinski_harabasz_score
ch_score = calinski_harabasz_score(X_scaled, y_pred)
print(f"Calinski-Harabasz指数: {ch_score:.2f}")
```


### 4.3 评估结果分析（以Iris数据集为例）
当 $K=3$ 时（与真实类别数一致），典型评估结果为：
- ARI ≈ 0.73（中等一致性，因K-means对非凸分布敏感，Iris中Versicolor和Virginica存在重叠）
- NMI ≈ 0.75（较好的互信息保留）
- 轮廓系数 ≈ 0.45（受簇重叠影响，未达到理想值）
- Calinski-Harabasz指数 ≈ 560（较大值，表明簇分离较好）


## 五、实例拓展-文档主题聚类应用

### 5.1 任务目标
使用 K-means 对文本数据进行聚类，自动发现文档主题。以 20 Newsgroups 数据集（包含不同主题的新闻组帖子）为例，实现从文本到主题的无监督学习。


### 5.2 技术路线
文本 → 预处理 → TF-IDF向量化 → K-means聚类 → 主题关键词提取


### 5.3 完整代码实现

#### 5.3.1 数据加载与预处理
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 加载数据集（选择4个主题类别）
categories = ['comp.graphics', 'sci.med', 'talk.politics.misc', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, 
                                remove=('headers', 'footers', 'quotes'),  # 去除无关信息
                                shuffle=True, random_state=42)
documents = newsgroups.data
true_labels = newsgroups.target
target_names = newsgroups.target_names

# 2. TF-IDF特征提取
tfidf = TfidfVectorizer(max_features=2000,  # 保留Top2000高频词
                        stop_words='english',  # 去除英文停用词（如the, is）
                        min_df=5)  # 仅保留至少出现在5个文档中的词
X_tfidf = tfidf.fit_transform(documents)
print(f"TF-IDF矩阵形状: {X_tfidf.shape}")  # (样本数, 特征数)
```

#### 5.3.2 K-means聚类与结果分析
```python
# 3. K-means聚类（假设已知主题数K=4）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(X_tfidf)

# 4. 提取每个簇的关键词（TF-IDF权重最高的词）
feature_names = tfidf.get_feature_names()
n_top_words = 10  # 每个簇取10个关键词
print("\n=== 每个簇的主题关键词 ===")
for i in range(4):
    # 获取簇中心对应的特征索引（按权重排序）
    cluster_centers = kmeans.cluster_centers_[i]
    top_indices = cluster_centers.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"簇{i}: {', '.join(top_words)}")

# 5. 聚类结果可视化（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())  # TF-IDF矩阵为稀疏矩阵，需转为数组

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_labels, cmap='viridis', alpha=0.6)
plt.xlabel('PCA特征1')
plt.ylabel('PCA特征2')
plt.title('文档主题聚类结果（TF-IDF+K-means）')
plt.colorbar(scatter, label='簇标签')
plt.show()
```


### 5.4 关键技术解释

1. **TF-IDF向量化**：将文本转换为数值特征，其中：
   - **TF（词频）**：词在文档中出现的频率
   - **IDF（逆文档频率）**：衡量词的重要性，罕见词IDF高
   - **TF-IDF = TF × IDF**：突出文档中具有区分度的关键词

2. **主题关键词提取**：K-means 簇中心的坐标对应每个词的 TF-IDF 权重，权重最高的词即为该簇的主题关键词（如“hockey, game, team”对应体育主题）。

3. **结果解读**：理想情况下，聚类标签应与真实主题（comp.graphics、sci.med等）对应，但由于文本语义复杂性，可能存在部分交叉（如“medical”和“health”可能同时出现在多个簇）。


## 六、思考题

1. **理论理解**：K-means 算法可能收敛到局部最优解，如何通过算法改进缓解这一问题？（至少列举2种方法）

2. **参数影响**：解释 `n_init` 和 `max_iter` 参数对 K-means 结果的影响，如何设置这两个参数平衡性能与效率？

3. **结果分析**：在 Iris 聚类实验中，若将 `n_clusters` 设置为 2，预期聚类结果会如何变化？SSE 和轮廓系数会增大还是减小？

4. **方法局限**：K-means 为什么不适合处理非凸形状的簇（如环形分布数据）？列举一种适合非凸簇的聚类算法。

5. **文本聚类**：在文档主题聚类中，若发现多个簇的关键词高度重叠，可能的原因是什么？如何优化？


## 七、代码填空答案

1. ①处填：3（Iris数据集包含3类鸢尾花，与真实类别数对应）  
2. ②处填：2（降至2维以便可视化）  
3. ③处填：'viridis'（或其他有效颜色映射，如'rainbow'、'plasma'）
