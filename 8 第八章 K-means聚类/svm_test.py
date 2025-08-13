# SVM实验代码测试
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def test_linear_svm():
    print("=== 线性SVM测试 ===")
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"数据集形状: {X.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"样本数量: {len(X)}")
    print(f"类别分布: {np.bincount(y)}")
    print()
    
    # 划分训练集与测试集(7:3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 数据标准化(SVM对特征尺度敏感)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集形状: {X_train_scaled.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")
    print(f"特征标准化后的均值: {X_train_scaled.mean():.4f}")
    print(f"特征标准化后的标准差: {X_train_scaled.std():.4f}")
    print()
    
    # 创建线性SVM模型
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    
    # 训练模型
    svm_linear.fit(X_train_scaled, y_train)
    
    # 预测测试集
    y_pred = svm_linear.predict(X_test_scaled)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 查看支持向量数量
    print(f"支持向量数量: {len(svm_linear.support_)}")
    print(f"决策函数系数形状: {svm_linear.coef_.shape}")
    print()
    
    return {
        'accuracy': accuracy,
        'support_vectors': len(svm_linear.support_),
        'coef_shape': svm_linear.coef_.shape
    }

def test_nonlinear_svm():
    print("=== 非线性SVM测试 ===")
    
    from sklearn.datasets import make_moons
    
    # 生成非线性数据(含噪声)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    print(f"数据集形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print()
    
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义核函数列表
    kernels = ['linear', 'rbf', 'poly']
    titles = ['Linear Kernel', 'RBF Kernel', 'Polynomial Kernel']
    accuracies = []
    
    # 训练不同核函数的SVM
    for kernel in kernels:
        svm = SVC(kernel=kernel, gamma='scale', C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"{kernel.upper()}核准确率: {accuracy:.4f}")
    
    best_kernel_idx = np.argmax(accuracies)
    best_kernel = kernels[best_kernel_idx].upper()
    best_accuracy = accuracies[best_kernel_idx]
    
    print(f"\n🏆 最佳核函数: {best_kernel} (准确率: {best_accuracy:.4f})")
    print()
    
    return {
        'linear_accuracy': accuracies[0],
        'rbf_accuracy': accuracies[1],
        'poly_accuracy': accuracies[2],
        'best_kernel': best_kernel,
        'best_accuracy': best_accuracy
    }

def test_parameter_optimization():
    print("=== 参数优化测试 ===")
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import make_classification
    
    # 生成分类数据集
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 网格搜索参数
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1.0],
        'kernel': ['rbf']
    }
    
    # 网格搜索
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"🔍 开始网格搜索...")
    print(f"🏆 最佳参数: {grid_search.best_params_}")
    print(f"📊 最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    # 测试集评估
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"🎯 调优后测试集准确率: {test_accuracy:.4f}")
    print()
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_accuracy
    }

def test_spam_classification():
    print("=== 垃圾邮件分类测试 ===")
    
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix
    
    # 加载新闻组数据集的子集
    categories = ['rec.sport.hockey', 'talk.politics.misc']
    
    # 加载训练和测试数据
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)
    
    print(f"📧 邮件数据集信息:")
    print(f"总邮件数量: {len(newsgroups_train.data) + len(newsgroups_test.data)}")
    print(f"训练集数量: {len(newsgroups_train.data)}")
    print(f"测试集数量: {len(newsgroups_test.data)}")
    print(f"类别名称: {newsgroups_train.target_names}")
    print()
    
    # TF-IDF特征化
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)
    X_test_tfidf = vectorizer.transform(newsgroups_test.data)
    
    print(f"🔤 TF-IDF特征化结果:")
    print(f"训练集特征维度: {X_train_tfidf.shape}")
    print(f"测试集特征维度: {X_test_tfidf.shape}")
    print(f"特征稀疏度: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")
    print()
    
    # 训练SVM分类器
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_classifier.fit(X_train_tfidf, newsgroups_train.target)
    
    # 预测
    y_pred = svm_classifier.predict(X_test_tfidf)
    
    # 计算准确率
    accuracy = accuracy_score(newsgroups_test.target, y_pred)
    print(f"📊 垃圾邮件分类结果:")
    print(f"测试集准确率: {accuracy:.4f}")
    print()
    
    # 详细分类报告
    print(f"📈 详细分类报告:")
    print(classification_report(newsgroups_test.target, y_pred, target_names=newsgroups_train.target_names))
    
    return {
        'accuracy': accuracy,
        'train_samples': len(newsgroups_train.data),
        'test_samples': len(newsgroups_test.data)
    }

if __name__ == "__main__":
    print("开始SVM实验测试...\n")
    
    # 测试线性SVM
    linear_results = test_linear_svm()
    
    # 测试非线性SVM
    nonlinear_results = test_nonlinear_svm()
    
    # 测试参数优化
    optimization_results = test_parameter_optimization()
    
    # 测试垃圾邮件分类
    spam_results = test_spam_classification()
    
    print("\n=== 实验总结 ===")
    print(f"线性SVM准确率: {linear_results['accuracy']:.4f}")
    print(f"最佳非线性核函数: {nonlinear_results['best_kernel']} (准确率: {nonlinear_results['best_accuracy']:.4f})")
    print(f"参数优化后准确率: {optimization_results['test_accuracy']:.4f}")
    print(f"垃圾邮件分类准确率: {spam_results['accuracy']:.4f}")