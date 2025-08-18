#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式决策树演示 - Google Colab优化版
专为在线Python平台设计的轻量级版本

使用方法:
1. 在Google Colab中运行此脚本
2. 使用交互式控件调整参数
3. 观察决策树训练结果和可视化
"""

# 安装必要的依赖包（仅在Colab中需要）
try:
    import google.colab
    print("🔍 检测到Google Colab环境，正在安装依赖包...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ipywidgets"])
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    print("💻 在本地Jupyter环境中运行")
except Exception as e:
    print(f"⚠️ 依赖安装可能失败: {e}")
    print("请确保已安装 ipywidgets: pip install ipywidgets")

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import time
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量存储数据和模型
GLOBAL_DATA = None
GLOBAL_MODEL = None
GLOBAL_RESULTS = {}

def load_watermelon_data():
    """加载西瓜数据集"""
    global GLOBAL_DATA
    
    # 西瓜数据集3.0α
    data = {
        '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', '乌黑', '青绿', 
               '浅白', '浅白', '青绿', '浅白', '乌黑', '浅白', '青绿'],
        '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '稍蜷', '硬挺',
               '硬挺', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '蜷缩', '蜷缩'],
        '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响', '沉闷', '清脆',
               '清脆', '浊响', '浊响', '沉闷', '浊响', '沉闷', '沉闷'],
        '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰', '稍糊', '清晰',
               '模糊', '模糊', '稍糊', '稍糊', '清晰', '模糊', '稍糊'],
        '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹', '稍凹', '平坦',
               '平坦', '平坦', '凹陷', '凹陷', '稍凹', '平坦', '稍凹'],
        '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', '硬滑', '软粘',
               '硬滑', '软粘', '硬滑', '硬滑', '软粘', '硬滑', '硬滑'],
        '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243,
               0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
        '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267,
                 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103],
        '好瓜': ['是', '是', '是', '是', '是', '是', '是', '是', '否', '否',
               '否', '否', '否', '否', '否', '否', '否']
    }
    
    df = pd.DataFrame(data)
    
    # 数据预处理
    encoders = {}
    feature_names = []
    
    # 对分类特征进行编码
    categorical_features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    for feature in categorical_features:
        encoder = LabelEncoder()
        df[f'{feature}_encoded'] = encoder.fit_transform(df[feature])
        encoders[feature] = encoder
        feature_names.append(feature)
        
    # 数值特征
    numerical_features = ['密度', '含糖率']
    feature_names.extend(numerical_features)
    
    # 目标变量编码
    target_encoder = LabelEncoder()
    df['好瓜_encoded'] = target_encoder.fit_transform(df['好瓜'])
    encoders['好瓜'] = target_encoder
    
    # 准备特征矩阵和目标向量
    feature_columns = [f'{f}_encoded' if f in categorical_features else f for f in feature_names]
    X = df[feature_columns].values
    y = df['好瓜_encoded'].values
    
    GLOBAL_DATA = {
        'df': df,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'encoders': encoders
    }
    
    return df

def train_decision_tree(max_depth=3, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
    """训练决策树模型"""
    global GLOBAL_MODEL, GLOBAL_RESULTS
    
    if GLOBAL_DATA is None:
        print("❌ 请先加载数据集！")
        return
    
    X, y = GLOBAL_DATA['X'], GLOBAL_DATA['y']
    feature_names = GLOBAL_DATA['feature_names']
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练模型
    start_time = time.time()
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 预测和评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # 保存结果
    GLOBAL_MODEL = model
    GLOBAL_RESULTS = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'actual_depth': model.get_depth(),
        'n_leaves': model.get_n_leaves(),
        'training_time': training_time,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }
    
    # 显示结果
    display_results()
    
def display_results():
    """显示训练结果"""
    if GLOBAL_MODEL is None or not GLOBAL_RESULTS:
        print("❌ 请先训练模型！")
        return
    
    results = GLOBAL_RESULTS
    model = GLOBAL_MODEL
    
    print("\n" + "="*50)
    print("🎯 决策树训练结果")
    print("="*50)
    print(f"📈 训练准确率: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)")
    print(f"📊 测试准确率: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"🌳 树的深度: {results['actual_depth']}")
    print(f"🍃 叶节点数: {results['n_leaves']}")
    print(f"⏱️ 训练时间: {results['training_time']:.4f} 秒")
    
    # 过拟合检查
    train_acc = results['train_accuracy']
    test_acc = results['test_accuracy']
    
    print("\n" + "-"*30)
    print("⚠️ 模型诊断")
    print("-"*30)
    
    if train_acc - test_acc > 0.15:
        print("🔴 警告: 可能过拟合！建议减少最大深度")
    elif train_acc < 0.7 and test_acc < 0.7:
        print("🟡 警告: 可能欠拟合！建议增加最大深度")
    else:
        print("✅ 模型拟合良好！")
    
    # 显示决策树结构（简化版）
    print("\n" + "-"*30)
    print("🌳 决策树结构（前3层）")
    print("-"*30)
    
    tree_text = export_text(
        model,
        feature_names=results['feature_names'],
        class_names=['坏瓜', '好瓜'],
        max_depth=3
    )
    print(tree_text)
    
    # 绘制可视化图表
    plot_results()
    
def plot_results():
    """绘制结果可视化"""
    if not GLOBAL_RESULTS:
        return
    
    results = GLOBAL_RESULTS
    model = GLOBAL_MODEL
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('决策树分析结果', fontsize=14, fontweight='bold')
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    categories = ['训练集', '测试集']
    accuracies = [results['train_accuracy'], results['test_accuracy']]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8)
    ax1.set_title('准确率对比')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1.1)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 特征重要性
    ax2 = axes[0, 1]
    importances = model.feature_importances_
    feature_names = results['feature_names']
    indices = np.argsort(importances)[::-1]
    
    ax2.bar(range(len(importances)), importances[indices], alpha=0.8, color='#2ecc71')
    ax2.set_title('特征重要性')
    ax2.set_ylabel('重要性')
    ax2.set_xticks(range(len(importances)))
    ax2.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    
    # 3. 类别分布
    ax3 = axes[1, 0]
    y = GLOBAL_DATA['y']
    class_counts = np.bincount(y)
    class_labels = ['坏瓜', '好瓜']
    colors = ['#e74c3c', '#2ecc71']
    
    ax3.pie(class_counts, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('数据集类别分布')
    
    # 4. 模型指标
    ax4 = axes[1, 1]
    metrics = ['深度', '叶节点', '时间(ms)']
    values = [
        results['actual_depth'],
        results['n_leaves'],
        results['training_time'] * 1000
    ]
    
    bars = ax4.bar(metrics, values, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8)
    ax4.set_title('模型复杂度')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{value:.1f}' if isinstance(value, float) else f'{value}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_interactive_demo():
    """创建交互式演示界面"""
    # 加载数据
    df = load_watermelon_data()
    
    print("🌳 交互式决策树演示")
    print("="*40)
    print(f"📊 数据集: 西瓜数据集 ({len(df)} 个样本, {len(GLOBAL_DATA['feature_names'])} 个特征)")
    print(f"🎯 目标: 预测西瓜好坏")
    print(f"📈 类别分布: 好瓜 {sum(GLOBAL_DATA['y'])} 个, 坏瓜 {len(GLOBAL_DATA['y']) - sum(GLOBAL_DATA['y'])} 个")
    print("\n调整下面的参数来训练不同的决策树模型：")
    
    # 创建交互式控件
    interactive_widget = interactive(
        train_decision_tree,
        max_depth=widgets.IntSlider(
            value=3, min=1, max=8, step=1,
            description='最大深度:', style={'description_width': 'initial'}
        ),
        min_samples_split=widgets.IntSlider(
            value=2, min=2, max=8, step=1,
            description='最小分裂样本:', style={'description_width': 'initial'}
        ),
        min_samples_leaf=widgets.IntSlider(
            value=1, min=1, max=6, step=1,
            description='最小叶节点样本:', style={'description_width': 'initial'}
        ),
        criterion=widgets.Dropdown(
            options=[('基尼系数', 'gini'), ('信息熵', 'entropy')],
            value='gini',
            description='分裂标准:', style={'description_width': 'initial'}
        )
    )
    
    return interactive_widget

def show_dataset_info():
    """显示数据集详细信息"""
    if GLOBAL_DATA is None:
        load_watermelon_data()
    
    df = GLOBAL_DATA['df']
    
    print("📋 西瓜数据集详细信息")
    print("="*40)
    print("\n前10行数据:")
    display(df.head(10))
    
    print("\n数据集统计信息:")
    print(f"• 总样本数: {len(df)}")
    print(f"• 特征数: {len(GLOBAL_DATA['feature_names'])}")
    print(f"• 分类特征: 色泽, 根蒂, 敲声, 纹理, 脐部, 触感")
    print(f"• 数值特征: 密度, 含糖率")
    print(f"• 目标变量: 好瓜 (是/否)")
    
    # 类别分布
    class_dist = df['好瓜'].value_counts()
    print(f"\n类别分布:")
    for class_name, count in class_dist.items():
        print(f"• {class_name}: {count} 个 ({count/len(df)*100:.1f}%)")

def show_usage_guide():
    """显示使用指南"""
    guide = """
    🎯 交互式决策树演示使用指南
    
    📋 主要功能:
    ✅ 基于西瓜数据集的决策树分类
    ✅ 交互式参数调节
    ✅ 实时训练和评估
    ✅ 可视化结果展示
    ✅ 过拟合/欠拟合检测
    
    🚀 快速开始:
    1. 运行 demo = create_interactive_demo() 创建交互界面
    2. 调整参数滑块观察不同设置的效果
    3. 查看训练结果和可视化图表
    
    💡 参数说明:
    • 最大深度: 限制树的深度，防止过拟合
    • 最小分裂样本: 节点分裂所需的最小样本数
    • 最小叶节点样本: 叶节点的最小样本数
    • 分裂标准: 基尼系数或信息熵
    
    📊 其他功能:
    • show_dataset_info(): 查看数据集详情
    • train_decision_tree(): 手动训练模型
    • plot_results(): 显示可视化结果
    
    ⚠️ 使用提示:
    • 数据集较小，建议参数不要过大
    • 观察训练/测试准确率差异
    • 尝试不同参数组合进行对比
    """
    print(guide)

# 主程序
if __name__ == "__main__":
    print("🌳 交互式决策树演示 - Google Colab版")
    print("="*50)
    
    # 显示使用指南
    show_usage_guide()
    
    print("\n🚀 开始使用:")
    print("-"*20)
    print("# 创建交互式演示")
    print("demo = create_interactive_demo()")
    print("display(demo)")
    print("\n# 查看数据集信息")
    print("show_dataset_info()")
    
    # 在Colab中自动启动
    try:
        import google.colab
        print("\n🔍 检测到Google Colab环境，自动启动演示...")
        demo = create_interactive_demo()
        display(demo)
    except ImportError:
        print("\n💻 请手动运行上述命令开始演示")