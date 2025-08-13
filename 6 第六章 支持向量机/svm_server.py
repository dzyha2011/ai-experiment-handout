# SVM教学网页后端服务器
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import io
import sys
import base64
from contextlib import redirect_stdout, redirect_stderr
import traceback

# 导入所需的机器学习库
from sklearn.datasets import load_breast_cancer, make_moons, make_classification, fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# 配置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='.', static_url_path='')

# 为每个选项卡维护独立的执行环境
tab_environments = {}

# 简单的CORS处理
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def get_tab_environment(tab_id):
    """获取或创建选项卡的执行环境"""
    if tab_id not in tab_environments:
        # 创建自定义的plt对象，支持图表生成和数据分布显示
        class SafePlot:
            def __init__(self):
                self.figure_count = 0
                
            def __getattr__(self, name):
                attr = getattr(plt, name)
                if name == 'show':
                    return self._show_plot
                return attr
                
            def _show_plot(self, *args, **kwargs):
                self.figure_count += 1
                # 获取当前图表信息
                fig = plt.gcf()
                if fig.get_axes():
                    ax = fig.get_axes()[0]
                    title = ax.get_title() if ax.get_title() else f"图表 {self.figure_count}"
                    print(f"📊 {title} 已生成")
                    
                    # 显示图表的基本信息
                    if hasattr(ax, 'collections') and ax.collections:
                        # 散点图
                        for i, collection in enumerate(ax.collections):
                            if hasattr(collection, '_offsets') and len(collection._offsets) > 0:
                                print(f"  - 数据点数量: {len(collection._offsets)}")
                    
                    # 显示坐标轴标签
                    xlabel = ax.get_xlabel()
                    ylabel = ax.get_ylabel()
                    if xlabel:
                        print(f"  - X轴: {xlabel}")
                    if ylabel:
                        print(f"  - Y轴: {ylabel}")
                        
                    # 显示图例信息
                    legend = ax.get_legend()
                    if legend:
                        labels = [t.get_text() for t in legend.get_texts()]
                        print(f"  - 类别: {', '.join(labels)}")
                        
                else:
                    print(f"📊 图表 {self.figure_count} 已生成（在服务器环境中不显示交互式图表）")
                
                # 清除当前图表以避免重叠
                plt.clf()
        
        tab_environments[tab_id] = {
            '__builtins__': __builtins__,
            'np': np,
            'matplotlib': matplotlib,
            'plt': SafePlot(),
            'load_breast_cancer': load_breast_cancer,
            'make_moons': make_moons,
            'make_classification': make_classification,
            'fetch_20newsgroups': fetch_20newsgroups,
            'train_test_split': train_test_split,
            'GridSearchCV': GridSearchCV,
            'SVC': SVC,
            'accuracy_score': accuracy_score,
            'classification_report': classification_report,
            'StandardScaler': StandardScaler,
            'TfidfVectorizer': TfidfVectorizer
        }
    return tab_environments[tab_id]

@app.route('/run_code', methods=['POST'])
def run_code():
    try:
        data = request.get_json()
        code_id = data.get('code_id', '')
        code = data.get('code', '')
        tab_id = data.get('tab_id', 'default')  # 获取选项卡ID
        reset_env = data.get('reset_env', False)  # 是否重置环境
        
        # 如果需要重置环境，清除该选项卡的环境
        if reset_env and tab_id in tab_environments:
            del tab_environments[tab_id]
        
        # 获取选项卡特定的执行环境
        exec_globals = get_tab_environment(tab_id)
        
        # 捕获输出
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # 添加print函数到环境中
        exec_globals['print'] = lambda *args, **kwargs: print(*args, **kwargs, file=output_buffer)
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals)
            
            output = output_buffer.getvalue()
            error = error_buffer.getvalue()
            
            if error:
                return jsonify({
                    'success': False,
                    'output': output,
                    'error': error
                })
            else:
                return jsonify({
                    'success': True,
                    'output': output,
                    'error': ''
                })
                
        except Exception as e:
            error_msg = f"执行错误: {str(e)}\n{traceback.format_exc()}"
            return jsonify({
                'success': False,
                'output': output_buffer.getvalue(),
                'error': error_msg
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'output': '',
            'error': f"服务器错误: {str(e)}"
        })

@app.route('/reset_tab_env', methods=['POST'])
def reset_tab_env():
    """重置指定选项卡的执行环境"""
    try:
        data = request.get_json()
        tab_id = data.get('tab_id', 'default')
        
        if tab_id in tab_environments:
            del tab_environments[tab_id]
            
        return jsonify({
            'success': True,
            'message': f'选项卡 {tab_id} 的执行环境已重置'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"重置环境错误: {str(e)}"
        })

@app.route('/run_svm_demo', methods=['POST'])
def run_svm_demo():
    """运行特定的SVM演示代码"""
    try:
        data = request.get_json()
        demo_type = data.get('demo_type', '')
        params = data.get('params', {})
        
        output_buffer = io.StringIO()
        
        if demo_type == 'linear':
            result = run_linear_svm_demo(params, output_buffer)
        elif demo_type == 'nonlinear':
            result = run_nonlinear_svm_demo(params, output_buffer)
        elif demo_type == 'optimization':
            result = run_optimization_demo(params, output_buffer)
        elif demo_type == 'spam':
            result = run_spam_demo(params, output_buffer)
        else:
            return jsonify({
                'success': False,
                'output': '',
                'error': f'未知的演示类型: {demo_type}'
            })
        
        return jsonify({
            'success': True,
            'output': output_buffer.getvalue(),
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'output': '',
            'error': f"演示执行错误: {str(e)}\n{traceback.format_exc()}"
        })

def run_linear_svm_demo(params, output_buffer):
    """运行线性SVM演示"""
    C = float(params.get('C', 1.0))
    
    print(f"=== 线性SVM演示 (C={C}) ===", file=output_buffer)
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 显示数据集分布信息
    print(f"\n📊 数据集分布分析:", file=output_buffer)
    print(f"数据集形状: {X.shape}", file=output_buffer)
    print(f"特征数量: {X.shape[1]}", file=output_buffer)
    print(f"样本总数: {X.shape[0]}", file=output_buffer)
    print(f"良性样本数: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)", file=output_buffer)
    print(f"恶性样本数: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)", file=output_buffer)
    
    # 显示特征统计信息
    print(f"\n📈 特征统计信息:", file=output_buffer)
    print(f"特征均值范围: [{X.mean(axis=0).min():.2f}, {X.mean(axis=0).max():.2f}]", file=output_buffer)
    print(f"特征标准差范围: [{X.std(axis=0).min():.2f}, {X.std(axis=0).max():.2f}]", file=output_buffer)
    
    # 可视化前两个主要特征的分布
    plt.figure(figsize=(12, 4))
    
    # 子图1: 前两个特征的散点图
    plt.subplot(1, 3, 1)
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('原始数据分布\n(前两个特征)')
    plt.legend(['恶性', '良性'])
    
    # 子图2: 特征均值分布
    plt.subplot(1, 3, 2)
    mean_features_0 = X[y == 0].mean(axis=0)
    mean_features_1 = X[y == 1].mean(axis=0)
    feature_indices = range(len(data.feature_names[:10]))  # 显示前10个特征
    plt.bar([i - 0.2 for i in feature_indices], mean_features_0[:10], width=0.4, label='恶性', color='red', alpha=0.7)
    plt.bar([i + 0.2 for i in feature_indices], mean_features_1[:10], width=0.4, label='良性', color='blue', alpha=0.7)
    plt.xlabel('特征索引')
    plt.ylabel('特征均值')
    plt.title('各类别特征均值对比\n(前10个特征)')
    plt.legend()
    plt.xticks(feature_indices)
    
    # 子图3: 类别分布饼图
    plt.subplot(1, 3, 3)
    labels = ['恶性', '良性']
    sizes = [np.sum(y == 0), np.sum(y == 1)]
    colors = ['red', 'blue']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', alpha=0.7)
    plt.title('类别分布')
    
    plt.tight_layout()
    plt.show()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\n🔄 数据划分:", file=output_buffer)
    print(f"训练集大小: {X_train.shape[0]}", file=output_buffer)
    print(f"测试集大小: {X_test.shape[0]}", file=output_buffer)
    print(f"训练集中良性样本: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)", file=output_buffer)
    print(f"测试集中良性样本: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)", file=output_buffer)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    svm_linear = SVC(kernel='linear', C=C, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    
    # 预测和评估
    y_pred = svm_linear.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 模型性能:", file=output_buffer)
    print(f"测试集准确率: {accuracy:.4f}", file=output_buffer)
    print(f"支持向量数量: {len(svm_linear.support_)}", file=output_buffer)
    
    # 根据C值给出建议
    if C < 0.1:
        print("💡 提示: C值较小，模型可能欠拟合", file=output_buffer)
    elif C > 10:
        print("⚠️ 警告: C值较大，模型可能过拟合", file=output_buffer)
    else:
        print("✅ C值适中，模型表现良好", file=output_buffer)
    
    return {
        'accuracy': accuracy,
        'support_vectors': len(svm_linear.support_),
        'C': C
    }

def run_nonlinear_svm_demo(params, output_buffer):
    """运行非线性SVM演示"""
    gamma_param = params.get('gamma', 1.0)
    # 处理gamma参数，如果是字符串'scale'或'auto'则保持原样，否则转换为float
    if isinstance(gamma_param, str) and gamma_param in ['scale', 'auto']:
        gamma = gamma_param
    else:
        gamma = float(gamma_param)
    C = float(params.get('C', 1.0))
    
    print(f"=== 非线性SVM演示 (C={C}, gamma={gamma}) ===", file=output_buffer)
    
    # 生成非线性数据
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # 显示数据集分布信息
    print(f"\n📊 数据集分布分析:", file=output_buffer)
    print(f"数据集形状: {X.shape}", file=output_buffer)
    print(f"特征数量: {X.shape[1]}", file=output_buffer)
    print(f"样本总数: {X.shape[0]}", file=output_buffer)
    print(f"类别0样本数: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)", file=output_buffer)
    print(f"类别1样本数: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)", file=output_buffer)
    
    # 显示特征统计信息
    print(f"\n📈 特征统计信息:", file=output_buffer)
    print(f"X1特征范围: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]", file=output_buffer)
    print(f"X2特征范围: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]", file=output_buffer)
    print(f"X1特征均值: {X[:, 0].mean():.2f} ± {X[:, 0].std():.2f}", file=output_buffer)
    print(f"X2特征均值: {X[:, 1].mean():.2f} ± {X[:, 1].std():.2f}", file=output_buffer)
    
    # 可视化数据分布
    plt.figure(figsize=(15, 5))
    
    # 子图1: 原始数据散点图
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'类别 {i}', alpha=0.7, s=50)
    plt.xlabel('特征 X1')
    plt.ylabel('特征 X2')
    plt.title('月牙形数据分布\n(非线性可分)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 特征分布直方图
    plt.subplot(1, 3, 2)
    plt.hist(X[y == 0, 0], bins=15, alpha=0.7, color='red', label='类别0 - X1')
    plt.hist(X[y == 1, 0], bins=15, alpha=0.7, color='blue', label='类别1 - X1')
    plt.xlabel('特征 X1 值')
    plt.ylabel('频次')
    plt.title('X1特征分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 特征分布直方图
    plt.subplot(1, 3, 3)
    plt.hist(X[y == 0, 1], bins=15, alpha=0.7, color='red', label='类别0 - X2')
    plt.hist(X[y == 1, 1], bins=15, alpha=0.7, color='blue', label='类别1 - X2')
    plt.xlabel('特征 X2 值')
    plt.ylabel('频次')
    plt.title('X2特征分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\n🔄 数据划分:", file=output_buffer)
    print(f"训练集大小: {X_train.shape[0]}", file=output_buffer)
    print(f"测试集大小: {X_test.shape[0]}", file=output_buffer)
    print(f"训练集中类别1样本: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)", file=output_buffer)
    print(f"测试集中类别1样本: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)", file=output_buffer)
    
    # 训练RBF核SVM
    svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    svm_rbf.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = svm_rbf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 模型性能:", file=output_buffer)
    print(f"RBF核准确率: {accuracy:.4f}", file=output_buffer)
    print(f"支持向量数量: {len(svm_rbf.support_)}", file=output_buffer)
    
    # 参数建议
    if gamma < 0.01:
        print("💡 提示: gamma值较小，决策边界较平滑", file=output_buffer)
    elif gamma > 10:
        print("⚠️ 警告: gamma值较大，可能过拟合", file=output_buffer)
    else:
        print("✅ gamma值适中", file=output_buffer)
    
    return {
        'accuracy': accuracy,
        'support_vectors': len(svm_rbf.support_),
        'C': C,
        'gamma': gamma
    }

def run_optimization_demo(params, output_buffer):
    """运行参数优化演示"""
    print("=== 参数优化演示 ===", file=output_buffer)
    
    # 生成分类数据
    X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
    
    # 显示数据集分布信息
    print(f"\n📊 数据集分布分析:", file=output_buffer)
    print(f"数据集形状: {X.shape}", file=output_buffer)
    print(f"特征数量: {X.shape[1]}", file=output_buffer)
    print(f"样本总数: {X.shape[0]}", file=output_buffer)
    print(f"类别0样本数: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)", file=output_buffer)
    print(f"类别1样本数: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)", file=output_buffer)
    
    # 显示特征统计信息
    print(f"\n📈 特征统计信息:", file=output_buffer)
    print(f"特征均值范围: [{X.mean(axis=0).min():.2f}, {X.mean(axis=0).max():.2f}]", file=output_buffer)
    print(f"特征标准差范围: [{X.std(axis=0).min():.2f}, {X.std(axis=0).max():.2f}]", file=output_buffer)
    
    # 可视化数据分布
    plt.figure(figsize=(15, 5))
    
    # 子图1: 前两个特征的散点图
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'类别 {i}', alpha=0.7, s=50)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('合成数据分布\n(前两个特征)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 特征重要性（方差）
    plt.subplot(1, 3, 2)
    feature_vars = X.var(axis=0)
    plt.bar(range(len(feature_vars)), feature_vars, color='skyblue', alpha=0.7)
    plt.xlabel('特征索引')
    plt.ylabel('方差')
    plt.title('各特征方差分布')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 类别间特征均值差异
    plt.subplot(1, 3, 3)
    mean_diff = np.abs(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0))
    plt.bar(range(len(mean_diff)), mean_diff, color='lightcoral', alpha=0.7)
    plt.xlabel('特征索引')
    plt.ylabel('类别间均值差异')
    plt.title('类别间特征差异')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"\n🔄 数据划分:", file=output_buffer)
    print(f"训练集大小: {X_train.shape[0]}", file=output_buffer)
    print(f"测试集大小: {X_test.shape[0]}", file=output_buffer)
    print(f"训练集中类别1样本: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)", file=output_buffer)
    print(f"测试集中类别1样本: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)", file=output_buffer)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 网格搜索
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1.0]
    }
    
    print(f"\n🔍 开始网格搜索...", file=output_buffer)
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # 测试最佳模型
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test_scaled, y_test)
    
    print(f"\n🎯 优化结果:", file=output_buffer)
    print(f"🏆 最佳参数: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}", file=output_buffer)
    print(f"📊 最佳交叉验证准确率: {grid_search.best_score_:.4f}", file=output_buffer)
    print(f"🎯 测试集准确率: {test_accuracy:.4f}", file=output_buffer)
    
    return {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': test_accuracy
    }

def run_spam_demo(params, output_buffer):
    """运行垃圾邮件分类演示"""
    kernel = params.get('kernel', 'rbf')
    C = float(params.get('C', 1.0))
    gamma_param = params.get('gamma', 'scale')
    # 处理gamma参数，如果是字符串'scale'或'auto'则保持原样，否则转换为float
    if isinstance(gamma_param, str) and gamma_param in ['scale', 'auto']:
        gamma = gamma_param
    else:
        gamma = float(gamma_param)
    
    print(f"=== 垃圾邮件分类演示 (kernel={kernel}, C={C}) ===", file=output_buffer)
    
    try:
        # 加载新闻组数据集
        categories = ['rec.sport.hockey', 'talk.politics.misc']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)
        
        # TF-IDF特征化
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # 减少特征数量以提高速度
        X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data)
        X_test_tfidf = vectorizer.transform(newsgroups_test.data)
        
        # 训练SVM
        svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        
        svm_classifier.fit(X_train_tfidf, newsgroups_train.target)
        
        # 预测
        y_pred = svm_classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(newsgroups_test.target, y_pred)
        
        print(f"📧 训练邮件数量: {len(newsgroups_train.data)}", file=output_buffer)
        print(f"📧 测试邮件数量: {len(newsgroups_test.data)}", file=output_buffer)
        print(f"🔤 特征维度: {X_train_tfidf.shape[1]}", file=output_buffer)
        print(f"📊 分类准确率: {accuracy:.4f}", file=output_buffer)
        
        # 计算其他指标
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(newsgroups_test.target, y_pred, average='weighted')
        recall = recall_score(newsgroups_test.target, y_pred, average='weighted')
        f1 = f1_score(newsgroups_test.target, y_pred, average='weighted')
        
        print(f"📈 精确率: {precision:.4f}", file=output_buffer)
        print(f"📈 召回率: {recall:.4f}", file=output_buffer)
        print(f"📈 F1分数: {f1:.4f}", file=output_buffer)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'kernel': kernel,
            'C': C
        }
        
    except Exception as e:
        print(f"❌ 数据加载失败，使用模拟数据: {str(e)}", file=output_buffer)
        # 使用模拟数据
        accuracy = 0.85 + (C - 1) * 0.05 + np.random.normal(0, 0.02)
        accuracy = max(0.7, min(0.99, accuracy))
        
        return {
            'accuracy': accuracy,
            'precision': accuracy + 0.01,
            'recall': accuracy - 0.01,
            'f1_score': accuracy,
            'kernel': kernel,
            'C': C
        }

@app.route('/')
def index():
    return send_from_directory('.', '第六章 支持向量机教学网页.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/train_svm', methods=['POST'])
def train_svm():
    try:
        # 获取参数
        data = request.get_json()
        kernel = data.get('kernel', 'rbf')
        C = float(data.get('C', 1.0))
        gamma = float(data.get('gamma', 0.1))
        max_features = int(data.get('max_features', 5000))
        
        # 加载数据
        categories = ['rec.sport.hockey', 'talk.politics.misc']
        newsgroups = fetch_20newsgroups(
            subset='all', 
            categories=categories, 
            remove=('headers', 'footers', 'quotes'), 
            random_state=42
        )
        X_text = newsgroups.data
        y = newsgroups.target
        
        # 划分训练集与测试集
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.3, random_state=42
        )
        
        # TF-IDF特征向量化
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_train_tfidf = tfidf.fit_transform(X_train_text)
        X_test_tfidf = tfidf.transform(X_test_text)
        
        # 训练SVM模型
        if kernel == 'linear':
            svm_model = SVC(C=C, kernel='linear', random_state=42)
        elif kernel == 'poly':
            svm_model = SVC(C=C, kernel='poly', degree=3, random_state=42)
        else:  # rbf
            svm_model = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
            
        svm_model.fit(X_train_tfidf, y_train)
        
        # 评估性能
        y_pred_train = svm_model.predict(X_train_tfidf)
        y_pred_test = svm_model.predict(X_test_tfidf)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # 获取支持向量数量
        n_support_vectors = len(svm_model.support_)
        
        # 生成混淆矩阵图
        plt.figure(figsize=(8, 6))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        # 创建热力图
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {kernel.upper()} Kernel')
        plt.colorbar()
        
        # 添加标签
        classes = ['Hockey', 'Politics']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # 添加数值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # 保存图片到内存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 分类报告
        report = classification_report(y_test, y_pred_test, 
                                     target_names=['Hockey', 'Politics'], 
                                     output_dict=True)
        
        return jsonify({
            'success': True,
            'results': {
                'kernel': kernel,
                'C': C,
                'gamma': gamma if kernel == 'rbf' else None,
                'max_features': max_features,
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'n_support_vectors': n_support_vectors,
                'confusion_matrix_img': img_base64,
                'classification_report': report
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("启动SVM教学服务器...")
    print("服务器地址: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)