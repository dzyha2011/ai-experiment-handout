#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLE实验讲义代码测试文件
测试所有代码片段是否可以正常运行
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, load_digits, fetch_olivetti_faces
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
warnings.filterwarnings('ignore')

def test_swiss_roll_example():
    """测试瑞士卷数据集降维实验"""
    print("=== 测试瑞士卷数据集降维实验 ===")
    
    # 导入必要库
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll
    from sklearn.manifold import LocallyLinearEmbedding
    from mpl_toolkits.mplot3d import Axes3D

    # 生成瑞士卷数据
    X, color = make_swiss_roll(n_samples=1500, random_state=42)

    # 1. 创建LLE模型（填空1：设置n_neighbors=12, n_components=2）
    lle = LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=42)

    # 2. 拟合模型并转换数据（填空2：调用fit_transform方法）
    X_lle = lle.fit_transform(X)

    # 可视化结果
    fig = plt.figure(figsize=(12, 6))

    # 绘制原始3D数据
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis')
    ax1.set_title('Original Swiss Roll Data')
    ax1.view_init(azim=120, elev=10)

    # 3. 绘制LLE降维后2D数据（填空3：补充X_lle的两个维度）
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis')
    ax2.set_title('LLE Reduced Data (2D)')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')

    plt.tight_layout()
    plt.savefig('j:\\实验讲义\\局部线性嵌入\\swiss_roll_lle_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 4. 输出重构误差（填空4：获取reconstruction_error_属性）
    print(f"Reconstruction error: {lle.reconstruction_error_:.4f}")
    print("瑞士卷数据集测试完成！\n")
    
    return True

def test_digits_example():
    """测试手写数字数据集降维实验"""
    print("=== 测试手写数字数据集降维实验 ===")
    
    # 导入必要库
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # 加载手写数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset shape: {X.shape}, Class distribution: {np.bincount(y)}")

    # 数据预处理：归一化
    X = X / X.max()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 5. 创建LLE模型（填空5：设置n_neighbors=15, n_components=20）
    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=20, random_state=42)

    # 6. 拟合训练数据并转换（填空6：调用fit_transform和transform方法）
    X_train_lle = lle.fit_transform(X_train)
    X_test_lle = lle.transform(X_test)

    # 使用KNN进行分类（降维前后对比）
    knn_orig = KNeighborsClassifier(n_neighbors=5)
    knn_orig.fit(X_train, y_train)
    y_pred_orig = knn_orig.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)

    # 7. 在降维数据上训练KNN（填空7：使用X_train_lle训练）
    knn_lle = KNeighborsClassifier(n_neighbors=5)
    knn_lle.fit(X_train_lle, y_train)
    y_pred_lle = knn_lle.predict(X_test_lle)
    acc_lle = accuracy_score(y_test, y_pred_lle)

    print(f"Original data accuracy: {acc_orig:.4f}")
    print(f"LLE reduced data accuracy: {acc_lle:.4f}")
    print(f"Feature dimension reduced from {X.shape[1]} to {X_train_lle.shape[1]}")

    # 可视化降维结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_lle[:, 0], X_train_lle[:, 1], c=y_train, 
                         cmap=plt.cm.get_cmap('tab10', 10), alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.title('LLE Visualization of Digits Dataset (2D)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('j:\\实验讲义\\局部线性嵌入\\digits_lle_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("手写数字数据集测试完成！\n")
    return True

def test_image_retrieval_system():
    """测试图像检索系统"""
    print("=== 测试图像检索系统 ===")
    
    try:
        # 导入必要库
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import LocallyLinearEmbedding
        from sklearn.datasets import fetch_olivetti_faces
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        from sklearn.decomposition import PCA
        import time

        # 设置随机种子确保结果可重现
        np.random.seed(42)

        class ImageRetrievalSystem:
            """基于LLE的图像检索系统"""
            
            def __init__(self, n_neighbors=15, n_components=50):
                self.n_neighbors = n_neighbors
                self.n_components = n_components
                self.lle = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, 
                    n_components=n_components, 
                    random_state=42
                )
                self.pca = PCA(n_components=n_components, random_state=42)
                self.features_original = None
                self.features_lle = None
                self.features_pca = None
                
            def fit(self, features):
                """训练降维模型"""
                print(f"Training on {features.shape[0]} images with {features.shape[1]} features...")
                
                # 保存原始特征
                self.features_original = features
                
                # LLE降维
                start_time = time.time()
                self.features_lle = self.lle.fit_transform(features)
                lle_time = time.time() - start_time
                
                # PCA降维（用于对比）
                start_time = time.time()
                self.features_pca = self.pca.fit_transform(features)
                pca_time = time.time() - start_time
                
                print(f"LLE降维时间: {lle_time:.3f}秒")
                print(f"PCA降维时间: {pca_time:.3f}秒")
                print(f"LLE重构误差: {self.lle.reconstruction_error_:.6f}")
                
            def retrieve_images(self, query_idx, method='lle', top_k=5):
                """图像检索"""
                if method == 'lle':
                    features = self.features_lle
                    query_feature = features[query_idx]
                elif method == 'pca':
                    features = self.features_pca
                    query_feature = features[query_idx]
                else:  # original
                    features = self.features_original
                    query_feature = features[query_idx]
                    
                # 计算相似度（使用余弦相似度）
                similarities = cosine_similarity([query_feature], features)[0]
                
                # 获取top-k相似图像索引（排除查询图像本身）
                similarities[query_idx] = -1  # 排除自身
                top_indices = similarities.argsort()[::-1][:top_k]
                
                return top_indices, similarities[top_indices]
            
            def evaluate_retrieval_performance(self, n_queries=10):  # 减少查询数量以加快测试
                """评估检索性能"""
                methods = ['original', 'lle', 'pca']
                results = {method: {'times': [], 'similarities': []} for method in methods}
                
                # 随机选择查询图像
                query_indices = np.random.choice(len(self.features_original), n_queries, replace=False)
                
                for query_idx in query_indices:
                    for method in methods:
                        start_time = time.time()
                        top_indices, similarities = self.retrieve_images(query_idx, method, top_k=5)
                        retrieval_time = time.time() - start_time
                        
                        results[method]['times'].append(retrieval_time)
                        results[method]['similarities'].append(np.mean(similarities))
                
                # 计算平均性能
                for method in methods:
                    avg_time = np.mean(results[method]['times'])
                    avg_similarity = np.mean(results[method]['similarities'])
                    print(f"{method.upper()} - 平均检索时间: {avg_time:.6f}秒, 平均相似度: {avg_similarity:.4f}")
                    
                return results

        # 加载Olivetti人脸数据集
        print("加载Olivetti人脸数据集...")
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X, y = faces.data, faces.target
        
        print(f"数据集信息: {X.shape[0]}张图像, 每张{X.shape[1]}维特征")
        print(f"共{len(np.unique(y))}个不同的人")
        
        # 数据预处理：标准化
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # 创建检索系统
        retrieval_system = ImageRetrievalSystem(n_neighbors=15, n_components=50)
        
        # 训练模型
        retrieval_system.fit(X)
        
        # 评估性能
        print("\n=== 检索性能评估 ===")
        performance_results = retrieval_system.evaluate_retrieval_performance(n_queries=10)
        
        # 简单的可视化测试
        query_idx = 10
        top_indices, similarities = retrieval_system.retrieve_images(query_idx, 'lle', top_k=5)
        print(f"\nLLE检索结果 - Top 5 相似度: {similarities}")
        
        print("图像检索系统测试完成！\n")
        return True
        
    except Exception as e:
        print(f"图像检索系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试LLE实验讲义中的所有代码...\n")
    
    test_results = []
    
    # 测试瑞士卷示例
    try:
        result1 = test_swiss_roll_example()
        test_results.append(("瑞士卷数据集", result1))
    except Exception as e:
        print(f"瑞士卷数据集测试失败: {e}")
        test_results.append(("瑞士卷数据集", False))
    
    # 测试手写数字示例
    try:
        result2 = test_digits_example()
        test_results.append(("手写数字数据集", result2))
    except Exception as e:
        print(f"手写数字数据集测试失败: {e}")
        test_results.append(("手写数字数据集", False))
    
    # 测试图像检索系统
    try:
        result3 = test_image_retrieval_system()
        test_results.append(("图像检索系统", result3))
    except Exception as e:
        print(f"图像检索系统测试失败: {e}")
        test_results.append(("图像检索系统", False))
    
    # 输出测试结果总结
    print("=== 测试结果总结 ===")
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    if all_passed:
        print("\n🎉 所有测试都通过了！代码可以正常运行。")
    else:
        print("\n⚠️ 部分测试失败，请检查相关代码。")
    
    return all_passed

if __name__ == "__main__":
    main()