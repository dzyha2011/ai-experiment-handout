import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import os

# 创建静态文件夹
if not os.path.exists('static'):
    os.makedirs('static')

print("开始生成SVM实验结果图...")

# 1. 线性SVM实验
print("\n1. 生成线性SVM结果图...")

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练线性SVM
svm_linear = LinearSVC(C=1.0, max_iter=10000, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# 预测并计算准确率
y_pred = svm_linear.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"线性SVM测试集准确率: {accuracy:.4f}")

# PCA降维可视化
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 绘制决策边界
h = 0.02
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 用降维后的数据训练模型
svm_vis = LinearSVC(C=1.0, max_iter=10000, random_state=42)
svm_vis.fit(X_train_pca, y_train)
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                     cmap=plt.cm.coolwarm, edgecolors='k', s=50)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.title('Linear SVM Decision Boundary (PCA-reduced Data)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/linear_svm_result.png', dpi=300, bbox_inches='tight')
plt.close()
print("线性SVM结果图已保存: static/linear_svm_result.png")

# 2. 非线性SVM实验
print("\n2. 生成非线性SVM结果图...")

# 生成非线性数据
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

# 不同核函数对比
kernels = ['linear', 'rbf', 'poly']
titles = ['Linear Kernel', 'RBF Kernel', 'Polynomial Kernel']
accuracies = []

plt.figure(figsize=(15, 5))
for i, kernel in enumerate(kernels):
    # 训练模型
    if kernel == 'poly':
        svm = SVC(kernel=kernel, gamma='scale', C=1.0, degree=3, random_state=42)
    else:
        svm = SVC(kernel=kernel, gamma='scale', C=1.0, random_state=42)
    
    svm.fit(X_train_moons, y_train_moons)
    
    # 计算准确率
    y_pred_kernel = svm.predict(X_test_moons)
    acc = accuracy_score(y_test_moons, y_pred_kernel)
    accuracies.append(acc)
    
    # 绘制决策边界
    plt.subplot(1, 3, i+1)
    h = 0.02
    x_min, x_max = X_moons[:, 0].min() - 1, X_moons[:, 0].max() + 1
    y_min, y_max = X_moons[:, 1].min() - 1, X_moons[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, 
                         cmap=plt.cm.coolwarm, edgecolors='k', s=50)
    plt.title(f'{titles[i]}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/nonlinear_svm_result.png', dpi=300, bbox_inches='tight')
plt.close()
print("非线性SVM结果图已保存: static/nonlinear_svm_result.png")
print(f"各核函数准确率: Linear={accuracies[0]:.3f}, RBF={accuracies[1]:.3f}, Poly={accuracies[2]:.3f}")

# 3. 参数调优实验
print("\n3. 生成参数调优结果图...")

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}

# 网格搜索
svm_grid = SVC(random_state=42)
grid_search = GridSearchCV(
    estimator=svm_grid, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search.fit(X_train_moons, y_train_moons)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 绘制参数调优热力图
results = pd.DataFrame(grid_search.cv_results_)
pivot = results.pivot(index='param_gamma', columns='param_C', values='mean_test_score')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.4f', 
            cbar_kws={'label': 'Cross-validation Accuracy'})
plt.title('Grid Search Accuracy Heatmap (RBF Kernel)', fontsize=14, fontweight='bold')
plt.xlabel('C (Penalty Parameter)', fontsize=12)
plt.ylabel('γ (Gamma Parameter)', fontsize=12)
plt.tight_layout()
plt.savefig('static/parameter_optimization.png', dpi=300, bbox_inches='tight')
plt.close()
print("参数调优结果图已保存: static/parameter_optimization.png")

# 4. 生成数据集可视化图
print("\n4. 生成数据集可视化图...")

# 原始月牙数据可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, 
                     cmap=plt.cm.coolwarm, edgecolors='k', s=60)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Nonlinear Dataset (Moons)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/moons_dataset.png', dpi=300, bbox_inches='tight')
plt.close()
print("月牙数据集可视化图已保存: static/moons_dataset.png")

# 5. 生成支持向量可视化
print("\n5. 生成支持向量可视化图...")

# 使用最佳参数训练模型
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_moons, y_train_moons)

# 绘制支持向量
plt.figure(figsize=(10, 6))
h = 0.02
x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# 绘制所有数据点
scatter = plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, 
                     cmap=plt.cm.coolwarm, edgecolors='k', s=50, alpha=0.7)

# 高亮支持向量
support_vectors = best_svm.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
           s=200, facecolors='none', edgecolors='red', linewidth=2, 
           label=f'Support Vectors ({len(support_vectors)})')

plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title(f'SVM with Support Vectors\nC={grid_search.best_params_["C"]}, γ={grid_search.best_params_["gamma"]}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/support_vectors.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"支持向量可视化图已保存: static/support_vectors.png")
print(f"支持向量数量: {len(support_vectors)}")

print("\n所有实验结果图生成完成！")
print("生成的文件列表:")
for file in os.listdir('static'):
    if file.endswith('.png'):
        print(f"  - static/{file}")

print("\n实验总结:")
print(f"1. 线性SVM在乳腺癌数据集上的准确率: {accuracy:.4f}")
print(f"2. 非线性SVM各核函数准确率:")
for i, kernel in enumerate(kernels):
    print(f"   - {kernel}: {accuracies[i]:.4f}")
print(f"3. 最佳参数组合: {grid_search.best_params_}")
print(f"4. 最佳交叉验证准确率: {grid_search.best_score_:.4f}")
print(f"5. 支持向量数量: {len(support_vectors)}")