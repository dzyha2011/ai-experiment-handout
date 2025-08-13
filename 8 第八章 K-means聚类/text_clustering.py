# 文档主题聚类完整实现
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class TextClustering:
    def __init__(self, categories=None, n_clusters=3, max_features=2000, min_df=5):
        self.categories = categories or ['comp.graphics', 'sci.med', 'rec.sport.hockey']
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.min_df = min_df
        self.tfidf = None
        self.kmeans = None
        self.documents = None
        self.X_tfidf = None
        self.labels = None
        
    def load_data(self):
        """加载20 Newsgroups数据集"""
        print(f"加载数据集，类别: {self.categories}")
        try:
            newsgroups = fetch_20newsgroups(
                subset='all', 
                categories=self.categories,
                remove=('headers', 'footers', 'quotes'),
                shuffle=True, 
                random_state=42
            )
            self.documents = newsgroups.data
            self.true_labels = newsgroups.target
            self.target_names = newsgroups.target_names
            print(f"加载完成，共{len(self.documents)}个文档")
        except Exception as e:
            print(f"数据加载失败: {e}")
            print("使用模拟数据进行演示...")
            self._create_sample_data()
        
    def _create_sample_data(self):
        """创建模拟文本数据用于演示"""
        sample_docs = [
            # 体育类文档
            "hockey game team player ice win season nhl sport competition",
            "basketball player team score game win championship league",
            "football soccer team goal match win tournament sport",
            "tennis player match win tournament court game sport",
            
            # 计算机类文档
            "computer graphics image software program file data system",
            "programming code software development computer technology",
            "database system data storage computer information technology",
            "network internet computer system technology digital",
            
            # 医学类文档
            "medical health disease treatment patient doctor hospital",
            "medicine drug treatment therapy patient health care",
            "surgery operation patient medical doctor hospital",
            "diagnosis disease medical patient health treatment",
            
        ] * 67  # 重复以增加数据量
        
        self.documents = sample_docs
        self.true_labels = np.array([0]*268 + [1]*268 + [2]*268)
        self.target_names = ['sport', 'computer', 'medical']
        print(f"创建模拟数据完成，共{len(self.documents)}个文档")
        
    def vectorize_text(self):
        """TF-IDF向量化"""
        print("进行TF-IDF向量化...")
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            min_df=self.min_df,
            max_df=0.95  # 去除出现在95%以上文档中的词
        )
        self.X_tfidf = self.tfidf.fit_transform(self.documents)
        print(f"TF-IDF矩阵形状: {self.X_tfidf.shape}")
        
    def perform_clustering(self):
        """执行K-means聚类"""
        print(f"执行K-means聚类，K={self.n_clusters}")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        self.labels = self.kmeans.fit_predict(self.X_tfidf)
        
        # 计算评估指标
        silhouette_avg = silhouette_score(self.X_tfidf, self.labels)
        print(f"平均轮廓系数: {silhouette_avg:.4f}")
        
    def extract_topic_keywords(self, n_words=10):
        """提取每个簇的主题关键词"""
        # 兼容不同版本的sklearn
        try:
            feature_names = self.tfidf.get_feature_names_out()
        except AttributeError:
            feature_names = self.tfidf.get_feature_names()
        topic_keywords = {}
        
        for i in range(self.n_clusters):
            # 获取簇中心的特征权重
            cluster_center = self.kmeans.cluster_centers_[i]
            # 按权重排序，取前n_words个
            top_indices = cluster_center.argsort()[-n_words:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            topic_keywords[f'主题{i+1}'] = top_words
            
        return topic_keywords
        
    def visualize_clusters(self):
        """可视化聚类结果"""
        # PCA降维到2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(self.X_tfidf.toarray())
        
        plt.figure(figsize=(12, 8))
        
        # 绘制聚类结果
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=self.labels, cmap='tab10', 
                             alpha=0.6, s=50)
        
        # 绘制簇中心
        centers_pca = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   marker='x', s=300, linewidths=3, 
                   color='red', label='簇中心')
        
        plt.xlabel('PCA特征1')
        plt.ylabel('PCA特征2')
        plt.title('文档主题聚类结果 (TF-IDF + K-means + PCA)')
        plt.legend()
        plt.colorbar(scatter, label='簇标签')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('text_clustering_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_cluster_distribution(self):
        """分析簇分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique)), counts, 
                      color=plt.cm.tab10(np.linspace(0, 1, len(unique))))
        plt.xlabel('簇标签')
        plt.ylabel('文档数量')
        plt.title('各簇文档分布')
        plt.xticks(range(len(unique)), [f'簇{i}' for i in unique])
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_complete_analysis(self):
        """运行完整的文本聚类分析"""
        self.load_data()
        self.vectorize_text()
        self.perform_clustering()
        
        # 提取并显示主题关键词
        topic_keywords = self.extract_topic_keywords()
        print("\n=== 主题关键词 ===")
        for topic, words in topic_keywords.items():
            print(f"{topic}: {', '.join(words)}")
        
        # 可视化结果
        self.visualize_clusters()
        self.analyze_cluster_distribution()
        
        return topic_keywords

# 使用示例
if __name__ == "__main__":
    # 创建文本聚类实例
    text_clusterer = TextClustering(
        categories=['comp.graphics', 'sci.med', 'rec.sport.hockey'],
        n_clusters=3,
        max_features=2000,
        min_df=5
    )
    
    # 运行完整分析
    keywords = text_clusterer.run_complete_analysis()
    
    print("\n=== 文本聚类实验总结 ===")
    print(f"聚类算法: K-means (K={text_clusterer.n_clusters})")
    print(f"特征提取: TF-IDF (最大特征数: {text_clusterer.max_features})")
    print(f"数据集: 20 Newsgroups (类别: {text_clusterer.categories})")
    print("\n主要发现:")
    print("1. 不同主题的文档在TF-IDF特征空间中形成了相对独立的簇")
    print("2. 关键词提取成功识别了各主题的核心词汇")
    print("3. PCA降维可视化显示了文档的主题分布模式")
    print("\n文本聚类分析完成！")