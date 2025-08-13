# AdaBoosting算法实验讲义

## 1. AdaBoosting算法原理

### 1.1 集成学习概述
集成学习（Ensemble Learning）通过组合多个弱分类器（Weak Classifier）的预测结果，构建一个强分类器（Strong Classifier）。弱分类器是指性能略优于随机猜测的模型（如简单决策树），通过特定策略组合后可获得高性能模型。

### 1.2 AdaBoosting核心思想
AdaBoosting（Adaptive Boosting）由Freund和Schapire于1997年提出，其核心思想包括：
- **适应性**：根据前一轮分类结果调整样本权重，错误分类样本获得更高权重
- **加权投票**：每个弱分类器根据其性能获得不同权重，性能好的分类器权重更高

### 1.3 数学原理

#### 1.3.1 样本权重更新
假设训练集为$ D=\{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}$，其中 $ y_i \in \{-1, +1\} $

初始化样本权重分布：
$ w_{1,i} = \frac{1}{N}, \quad i=1,2,...,N $

第$  t $ 轮迭代中，弱分类器 $h_t(x) $ 的分类错误率：
$ \epsilon_t = \frac{\sum_{i=1}^{N} w_{t,i} \cdot I(h_t(x_i) \neq y_i)}{\sum_{i=1}^{N} w_{t,i}} $

分类器权重：
$\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right) $

样本权重更新：
$w_{t+1,i} = \frac{w_{t,i} \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t} $
其中 $Z_t$ 为归一化因子，确保权重和为1。

#### 1.3.2 最终分类器
$ H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right) $

### 1.4 与其他集成方法对比
| 方法 | 核心思想 | 优势 | 劣势 |
|------|----------|------|------|
| AdaBoosting | 顺序训练，关注错误样本 | 简单高效，不易过拟合 | 对噪声和异常值敏感 |
| 随机森林 | 并行训练，特征随机采样 | 鲁棒性强，可并行 | 模型较复杂，解释性差 |
| GBDT | 基于梯度下降优化损失函数 | 性能优异，适用范围广 | 训练慢，参数敏感 |

## 2. AdaBoosting算法核心步骤

Adaboosting算法流程如下：

1. **初始化**：
   - 设置迭代次数 $T$
   - 初始化样本权重 $w_{1,i} = 1/N$

2. **迭代训练弱分类器**（for t=1 to T）：
   a. 基于权重 $w_t$ 训练弱分类器 $h_t(x) $
   b. 计算分类错误率 $ \epsilon_t $
   c. 计算分类器权重 $\alpha_t $
   d. 更新样本权重 $w_{t+1,i} $

3. **构建强分类器**：
   - 组合所有弱分类器：$H(x) = \text{sign}(\sum_{t=1}^{T} \alpha_t h_t(x)) $

4. **输出最终模型**

## 3. Python实现Adaboosting算法

### 3.1 实验目的
- 掌握Adaboost算法的基本实现
- 理解弱分类器数量对模型性能的影响
- 学习使用scikit-learn进行分类模型评估

### 3.2 实验环境
- Python 3.7+
- scikit-learn 0.24+
- numpy, pandas, matplotlib

### 3.3 数据集选择
使用scikit-learn内置的乳腺癌数据集（breast_cancer），包含569个样本，30个特征，二分类任务（良性/恶性）。

### 3.4 代码实现（含填空）

```python
# 导入必要的库
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# 加载数据集
data = load_breast_cancer()
X = data.data  # 特征矩阵
y = data.target  # 标签
feature_names = data.feature_names  # 特征名称

# 数据分割（填空1：补充test_size参数，建议值0.3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=____, random_state=42)

# 创建Adaboost模型（填空2：补充n_estimators参数，建议值50）
model = AdaBoostClassifier(n_estimators=____, random_state=42)

# 训练模型（填空3：补充训练数据）
model.fit(____, ____)

# 预测（填空4：补充测试数据）
y_pred = model.predict(____)

# 评估准确率（填空5：补充真实值和预测值）
accuracy = accuracy_score(____, ____)
print(f"模型准确率: {accuracy:.2f}")

# 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)
```

### 3.5 代码填空答案

1. test_size=0.3
2. n_estimators=50
3. model.fit(X_train, y_train)
4. y_pred = model.predict(X_test)
5. accuracy = accuracy_score(y_test, y_pred)## 5. 实例拓展-商品评论情感分析

## 4. AdaBoosting算法性能分析

### 4.1 弱分类器数量对性能的影响
当弱分类器数量（n_estimators）变化时，模型性能变化趋势：
- 初始阶段：随着n_estimators增加，准确率快速提升
- 稳定阶段：达到一定数量后，准确率趋于稳定
- 过拟合风险：数量过多可能导致过拟合测试集

### 4.2 学习率的影响
学习率（learning_rate）控制每个弱分类器的贡献：
- 较小学习率需要更多弱分类器，模型更稳健
- 较大学习率收敛快，但可能过拟合

### 4.3 性能指标解读
以乳腺癌数据集实验为例：
- 准确率（Accuracy）：整体分类正确率
- 精确率（Precision）：预测为恶性的样本中，实际为恶性的比例
- 召回率（Recall）：实际为恶性的样本中，被正确预测的比例
- F1分数：精确率和召回率的调和平均

### 4.4 可视化分析
建议绘制：
- 不同n_estimators下的准确率曲线
- ROC曲线及AUC值
- 特征重要性条形图（model.feature_importances_）

## 5. 实例拓展-商品评论情感分析

### 5.1 问题背景
商品评论情感分析旨在自动识别用户评论中的情感倾向（正面/负面），为商家提供产品改进依据和消费者反馈分析。

### 5.2 完整代码实现

```python
# 商品评论情感分析 - Adaboost实现
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# 1. 准备模拟数据
# 正面评论
positive_comments = [
    "这个产品非常好用，性价比很高，推荐购买！",
    "质量超出预期，使用体验很棒，五星好评！",
    "服务态度好，物流快，产品和描述一致",
    "已经是第二次购买了，家人都很喜欢",
    "功能齐全，操作简单，非常满意的一次购物"
]

# 负面评论
negative_comments = [
    "质量太差了，用了一天就坏了，不推荐购买",
    "与描述严重不符，客服态度恶劣，非常失望",
    "物流超级慢，包装破损，产品有瑕疵",
    "价格虚高，性能一般，后悔购买",
    "使用起来很不方便，体验感极差"
]

# 创建DataFrame
df_positive = pd.DataFrame({"comment": positive_comments, "label": 1})  # 1表示正面
df_negative = pd.DataFrame({"comment": negative_comments, "label": 0})  # 0表示负面
df = pd.concat([df_positive, df_negative], ignore_index=True)

# 2. 数据预处理
# 中文分词函数
def tokenize(text):
    # 使用jieba进行分词
    words = jieba.cut(text)
    # 返回空格分隔的词语
    return " ".join(words)

# 对评论进行分词处理
df["tokenized_comment"] = df["comment"].apply(tokenize)

# 3. 数据分割
# 特征与标签
X = df["tokenized_comment"]
y = df["label"]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 构建模型 pipeline
# 创建包含TF-IDF和Adaboost的管道
model = make_pipeline(
    TfidfVectorizer(max_features=1000),
    AdaBoostClassifier(base_estimator=MultinomialNB(), n_estimators=50, random_state=42)#使用朴素贝叶斯作为弱分类器
)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 预测与评估
# 在测试集上预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 7. 实际应用示例
def predict_sentiment(text):
    """预测单条评论的情感倾向"""
    # 分词处理
    tokenized_text = tokenize(text)
    # 预测情感概率
    prob = model.predict_proba([tokenized_text])[0][1]
    sentiment = "正面" if prob > 0.5 else "负面"
    return f"评论: {text}\n情感倾向: {sentiment} (概率: {prob:.2f})"

# 测试示例评论
test_comments = [
    "这个商品真的很好用，下次还会购买",
    "质量太差，浪费钱，不建议购买"
]

for comment in test_comments:
    print("\n" + predict_sentiment(comment))
```

### 5.3 应用场景
- 电商平台评论分析
- 舆情监控系统
- 产品口碑管理
- 客户满意度调查

### 5.4 思考题
1. 解释Adaboost中样本权重和分类器权重的更新机制，及其对模型性能的影响。
2. 当弱分类器数量增加时，Adaboost模型的偏差和方差如何变化？为什么？
3. 在商品评论情感分析中，如果数据存在严重的类别不平衡（如90%正面评论），如何调整Adaboost模型来提高少数类的识别率？
4. 比较Adaboost与随机森林在处理高维稀疏数据（如文本特征）时的优缺点。
5. 尝试推导Adaboost算法中分类器权重α_t的计算公式，并说明其意义。

### 5.5 注意事项

1. **环境依赖**：运行此代码需要安装jieba库，可通过`pip install jieba`命令安装
2. **数据规模**：实际应用中建议使用更大规模的标注数据集
3. **参数调优**：可通过网格搜索优化`max_features`、`n_estimators`等参数
4. **中文处理**：对于更复杂的中文文本，可考虑添加停用词过滤和词性筛选