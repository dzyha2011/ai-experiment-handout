from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 测试不同最大深度的影响
max_depths = range(1, 11)
train_scores = []
test_scores = []

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, 'o-', label='训练集准确率', linewidth=2)
plt.plot(max_depths, test_scores, 's-', label='测试集准确率', linewidth=2)
plt.xlabel('最大深度')
plt.ylabel('准确率')
plt.title('决策树深度对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(max_depths)
plt.savefig('results/pruning_depth_effect.png', dpi=300, bbox_inches='tight')
plt.close()

# 测试不同最小叶子节点样本数的影响
min_samples_leafs = [1, 2, 5, 10, 20, 30]
train_scores2 = []
test_scores2 = []

for min_leaf in min_samples_leafs:
    clf = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_scores2.append(accuracy_score(y_train, train_pred))
    test_scores2.append(accuracy_score(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(min_samples_leafs, train_scores2, 'o-', label='训练集准确率', linewidth=2)
plt.plot(min_samples_leafs, test_scores2, 's-', label='测试集准确率', linewidth=2)
plt.xlabel('最小叶子节点样本数')
plt.ylabel('准确率')
plt.title('叶子节点最小样本数对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(min_samples_leafs)
plt.savefig('results/pruning_leaf_effect.png', dpi=300, bbox_inches='tight')
plt.close()

# 生成决策树可视化图片
best_depth = max_depths[test_scores.index(max(test_scores))]
clf_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
clf_best.fit(X_train, y_train)

plt.figure(figsize=(20, 12))
plot_tree(clf_best, 
          feature_names=breast_cancer.feature_names,
          class_names=breast_cancer.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('乳腺癌数据集决策树可视化', fontsize=16, pad=20)
plt.savefig('results/iris_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

print('剪枝优化结果图片已保存到results文件夹')
print('决策树可视化图片已生成')
print(f'最佳深度测试准确率: {max(test_scores):.4f} (深度={max_depths[test_scores.index(max(test_scores))]})') 
print(f'最佳叶子节点数测试准确率: {max(test_scores2):.4f} (最小样本数={min_samples_leafs[test_scores2.index(max(test_scores2))]})')