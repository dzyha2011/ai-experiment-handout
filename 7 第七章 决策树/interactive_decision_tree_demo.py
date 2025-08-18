#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式决策树演示程序 - 在线Python平台版本
适用于Google Colab、Jupyter Notebook等在线Python环境

功能特性：
- 内置西瓜数据集
- 交互式参数调节
- 决策树可视化
- 性能评估和警告提示
- 无需外部服务器
"""

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
from ipywidgets import interact, interactive, fixed, interact_manual
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class InteractiveDecisionTreeDemo:
    def __init__(self):
        """初始化交互式决策树演示"""
        self.load_watermelon_dataset()
        self.setup_encoders()
        self.create_widgets()
        self.current_model = None
        self.current_results = {}
        
    def load_watermelon_dataset(self):
        """加载内置的西瓜数据集"""
        # 西瓜数据集3.0α
        data = {
            '编号': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
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
        
        self.df = pd.DataFrame(data)
        print("✅ 西瓜数据集加载成功！")
        print(f"数据集形状: {self.df.shape}")
        print(f"特征列: {list(self.df.columns[1:-1])}")
        print(f"目标变量: {self.df.columns[-1]}")
        print(f"类别分布: {self.df['好瓜'].value_counts().to_dict()}")
        
    def setup_encoders(self):
        """设置标签编码器"""
        self.encoders = {}
        self.feature_names = []
        
        # 对分类特征进行编码
        categorical_features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        for feature in categorical_features:
            encoder = LabelEncoder()
            self.df[f'{feature}_encoded'] = encoder.fit_transform(self.df[feature])
            self.encoders[feature] = encoder
            self.feature_names.append(feature)
            
        # 数值特征直接使用
        numerical_features = ['密度', '含糖率']
        self.feature_names.extend(numerical_features)
        
        # 目标变量编码
        target_encoder = LabelEncoder()
        self.df['好瓜_encoded'] = target_encoder.fit_transform(self.df['好瓜'])
        self.encoders['好瓜'] = target_encoder
        
        # 准备特征矩阵和目标向量
        feature_columns = [f'{f}_encoded' if f in categorical_features else f for f in self.feature_names]
        self.X = self.df[feature_columns].values
        self.y = self.df['好瓜_encoded'].values
        
        print("✅ 数据编码完成！")
        
    def create_widgets(self):
        """创建交互式控件"""
        # 参数控制滑块
        self.max_depth_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description='最大深度:',
            style={'description_width': 'initial'}
        )
        
        self.min_samples_split_slider = widgets.IntSlider(
            value=2,
            min=2,
            max=10,
            step=1,
            description='最小分裂样本数:',
            style={'description_width': 'initial'}
        )
        
        self.min_samples_leaf_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=8,
            step=1,
            description='最小叶节点样本数:',
            style={'description_width': 'initial'}
        )
        
        self.criterion_dropdown = widgets.Dropdown(
            options=[('基尼系数', 'gini'), ('信息熵', 'entropy')],
            value='gini',
            description='分裂标准:',
            style={'description_width': 'initial'}
        )
        
        # 训练按钮
        self.train_button = widgets.Button(
            description='🚀 训练决策树',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.train_button.on_click(self.on_train_button_clicked)
        
        # 输出区域
        self.output_area = widgets.Output()
        
        print("✅ 交互式控件创建完成！")
        
    def display_interface(self):
        """显示交互界面"""
        # 标题
        title = widgets.HTML(
            value="<h2 style='color: #2c3e50; text-align: center; margin-bottom: 20px;'>🌳 交互式决策树演示</h2>"
        )
        
        # 数据集信息
        dataset_info = widgets.HTML(
            value=f"""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <h4 style='color: #495057; margin-top: 0;'>📊 数据集信息</h4>
                <p><strong>数据集:</strong> 西瓜数据集3.0α</p>
                <p><strong>样本数:</strong> {len(self.df)} 个</p>
                <p><strong>特征数:</strong> {len(self.feature_names)} 个</p>
                <p><strong>类别分布:</strong> 好瓜: {sum(self.y)} 个, 坏瓜: {len(self.y) - sum(self.y)} 个</p>
            </div>
            """
        )
        
        # 参数控制面板
        params_box = widgets.VBox([
            widgets.HTML("<h4 style='color: #495057;'>⚙️ 模型参数</h4>"),
            self.max_depth_slider,
            self.min_samples_split_slider,
            self.min_samples_leaf_slider,
            self.criterion_dropdown,
            self.train_button
        ], layout=widgets.Layout(padding='15px', border='1px solid #dee2e6', border_radius='8px'))
        
        # 整体布局
        interface = widgets.VBox([
            title,
            dataset_info,
            params_box,
            self.output_area
        ])
        
        display(interface)
        
        # 初始训练
        self.train_decision_tree()
        
    def on_train_button_clicked(self, button):
        """训练按钮点击事件"""
        with self.output_area:
            clear_output(wait=True)
            self.train_decision_tree()
            
    def train_decision_tree(self):
        """训练决策树并显示结果"""
        with self.output_area:
            print("🔄 正在训练决策树...")
            
            # 获取参数
            max_depth = self.max_depth_slider.value
            min_samples_split = self.min_samples_split_slider.value
            min_samples_leaf = self.min_samples_leaf_slider.value
            criterion = self.criterion_dropdown.value
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
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
            
            # 保存当前模型和结果
            self.current_model = model
            self.current_results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'actual_depth': model.get_depth(),
                'n_leaves': model.get_n_leaves(),
                'training_time': training_time,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            clear_output(wait=True)
            self.display_results()
            
    def display_results(self):
        """显示训练结果"""
        if not self.current_model:
            return
            
        results = self.current_results
        
        # 结果摘要
        print("\n" + "="*60)
        print("🎯 训练结果摘要")
        print("="*60)
        print(f"📈 训练准确率: {results['train_accuracy']:.4f} ({results['train_accuracy']*100:.2f}%)")
        print(f"📊 测试准确率: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"🌳 实际深度: {results['actual_depth']}")
        print(f"🍃 叶节点数: {results['n_leaves']}")
        print(f"⏱️ 训练时间: {results['training_time']:.4f} 秒")
        
        # 过拟合/欠拟合警告
        self.check_overfitting_underfitting()
        
        # 显示决策树结构
        self.display_tree_structure()
        
        # 显示可视化图表
        self.display_visualizations()
        
    def check_overfitting_underfitting(self):
        """检查过拟合和欠拟合"""
        results = self.current_results
        train_acc = results['train_accuracy']
        test_acc = results['test_accuracy']
        
        print("\n" + "-"*40)
        print("⚠️ 模型诊断")
        print("-"*40)
        
        if train_acc - test_acc > 0.1:
            print("🔴 警告: 可能存在过拟合！")
            print("   建议: 减少最大深度或增加最小叶节点样本数")
        elif train_acc < 0.8 and test_acc < 0.8:
            print("🟡 警告: 可能存在欠拟合！")
            print("   建议: 增加最大深度或减少最小分裂样本数")
        else:
            print("✅ 模型拟合良好！")
            
    def display_tree_structure(self):
        """显示决策树结构"""
        print("\n" + "-"*40)
        print("🌳 决策树结构")
        print("-"*40)
        
        # 生成树的文本表示
        tree_text = export_text(
            self.current_model,
            feature_names=self.feature_names,
            class_names=['坏瓜', '好瓜'],
            max_depth=3  # 限制显示深度以避免过长
        )
        
        print(tree_text)
        
    def display_visualizations(self):
        """显示可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('决策树分析可视化', fontsize=16, fontweight='bold')
        
        # 1. 准确率对比
        ax1 = axes[0, 0]
        categories = ['训练集', '测试集']
        accuracies = [self.current_results['train_accuracy'], self.current_results['test_accuracy']]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
        ax1.set_title('训练集 vs 测试集准确率', fontweight='bold')
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 特征重要性
        ax2 = axes[0, 1]
        if hasattr(self.current_model, 'feature_importances_'):
            importances = self.current_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax2.bar(range(len(importances)), importances[indices], alpha=0.7, color='#2ecc71')
            ax2.set_title('特征重要性', fontweight='bold')
            ax2.set_ylabel('重要性')
            ax2.set_xticks(range(len(importances)))
            ax2.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        
        # 3. 类别分布
        ax3 = axes[1, 0]
        class_counts = np.bincount(self.y)
        class_labels = ['坏瓜', '好瓜']
        colors = ['#e74c3c', '#2ecc71']
        
        wedges, texts, autotexts = ax3.pie(class_counts, labels=class_labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('数据集类别分布', fontweight='bold')
        
        # 4. 模型复杂度指标
        ax4 = axes[1, 1]
        metrics = ['深度', '叶节点数', '训练时间(ms)']
        values = [
            self.current_results['actual_depth'],
            self.current_results['n_leaves'],
            self.current_results['training_time'] * 1000
        ]
        
        bars = ax4.bar(metrics, values, color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.7)
        ax4.set_title('模型复杂度指标', fontweight='bold')
        ax4.set_ylabel('数值')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.2f}' if isinstance(value, float) else f'{value}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    def show_dataset_sample(self):
        """显示数据集样本"""
        print("\n" + "="*60)
        print("📋 数据集样本预览")
        print("="*60)
        display(self.df.head(10))
        
    def show_feature_analysis(self):
        """显示特征分析"""
        print("\n" + "="*60)
        print("🔍 特征分析")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特征分布分析', fontsize=16, fontweight='bold')
        
        categorical_features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        
        for i, feature in enumerate(categorical_features):
            ax = axes[i//3, i%3]
            
            # 计算每个特征值对应的好瓜比例
            feature_counts = self.df.groupby([feature, '好瓜']).size().unstack(fill_value=0)
            feature_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], alpha=0.7)
            
            ax.set_title(f'{feature} 分布', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('数量')
            ax.legend(['坏瓜', '好瓜'])
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 数值特征分析
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('数值特征分布', fontsize=16, fontweight='bold')
        
        numerical_features = ['密度', '含糖率']
        
        for i, feature in enumerate(numerical_features):
            ax = axes[i]
            
            # 按类别分组绘制直方图
            good_melons = self.df[self.df['好瓜'] == '是'][feature]
            bad_melons = self.df[self.df['好瓜'] == '否'][feature]
            
            ax.hist(good_melons, alpha=0.7, label='好瓜', color='#2ecc71', bins=8)
            ax.hist(bad_melons, alpha=0.7, label='坏瓜', color='#e74c3c', bins=8)
            
            ax.set_title(f'{feature} 分布', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('频次')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

# 使用说明和示例代码
def show_usage_instructions():
    """显示使用说明"""
    instructions = """
    🎯 交互式决策树演示使用指南
    
    📋 功能特性:
    ✅ 内置西瓜数据集，无需外部文件
    ✅ 交互式参数调节（最大深度、最小分裂样本数等）
    ✅ 实时训练和性能评估
    ✅ 决策树结构可视化
    ✅ 过拟合/欠拟合警告
    ✅ 特征重要性分析
    ✅ 完全兼容Google Colab和Jupyter Notebook
    
    🚀 快速开始:
    1. 运行下面的代码创建演示实例
    2. 调整参数滑块
    3. 点击"训练决策树"按钮
    4. 查看结果和可视化图表
    
    💡 参数说明:
    • 最大深度: 控制树的最大深度，防止过拟合
    • 最小分裂样本数: 内部节点分裂所需的最小样本数
    • 最小叶节点样本数: 叶节点所需的最小样本数
    • 分裂标准: 基尼系数或信息熵
    
    ⚠️ 注意事项:
    • 数据集较小，建议参数不要设置过大
    • 观察训练集和测试集准确率差异，避免过拟合
    • 可以多次调整参数进行对比实验
    """
    
    print(instructions)

# 主程序入口
if __name__ == "__main__":
    # 显示使用说明
    show_usage_instructions()
    
    print("\n" + "="*60)
    print("🚀 正在初始化交互式决策树演示...")
    print("="*60)
    
    # 创建演示实例
    demo = InteractiveDecisionTreeDemo()
    
    print("\n" + "="*60)
    print("✅ 初始化完成！请使用以下命令开始演示:")
    print("="*60)
    print("\n# 显示交互界面")
    print("demo.display_interface()")
    print("\n# 查看数据集样本")
    print("demo.show_dataset_sample()")
    print("\n# 查看特征分析")
    print("demo.show_feature_analysis()")
    
    # 自动显示界面（在Colab中）
    try:
        import google.colab
        print("\n🔍 检测到Google Colab环境，自动显示交互界面...")
        demo.display_interface()
    except ImportError:
        print("\n💻 在Jupyter Notebook中，请手动运行 demo.display_interface() 显示界面")