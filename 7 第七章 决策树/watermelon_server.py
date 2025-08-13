from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import os
import io
import base64
from PIL import Image

app = Flask(__name__)
CORS(app)

# 全局变量存储数据和模型
watermelon_data = None
X_encoded = None
y_encoded = None
feature_names = None
label_encoders = {}
target_encoder = None

def load_watermelon_data():
    global watermelon_data, X_encoded, y_encoded, feature_names, label_encoders, target_encoder
    
    # 读取西瓜数据集
    watermelon_data = pd.read_csv('watermelon_3.csv')
    
    # 分离特征和目标
    X = watermelon_data.drop(['编号', '好瓜'], axis=1)
    y = watermelon_data['好瓜']
    
    feature_names = X.columns.tolist()
    
    # 对分类特征进行编码
    X_encoded = X.copy()
    label_encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object':  # 分类特征
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # 对目标变量编码
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X_encoded, y_encoded

@app.route('/train_tree', methods=['POST'])
def train_tree():
    try:
        params = request.json
        max_depth = params.get('max_depth', 3)
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        criterion = params.get('criterion', 'gini')
        
        # 训练决策树
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.3, random_state=42
        )
        
        # 记录训练时间
        import time
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 计算准确率
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 使用export_graphviz生成DOT格式的决策树可视化
        dot_data = export_graphviz(
            clf,
            feature_names=feature_names,
            class_names=target_encoder.classes_,
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None  # 返回字符串而不是写入文件
        )
        
        # 尝试使用graphviz生成SVG（如果可用）
        svg_content = None
        img_base64 = None
        
        try:
            import graphviz
            graph = graphviz.Source(dot_data)
            svg_content = graph.pipe(format='svg').decode('utf-8')
            print("Graphviz SVG可视化生成成功")
        except Exception as e:
            print(f"Graphviz SVG生成失败: {e}")
            print("将使用DOT格式文本显示决策树结构")
            # 如果Graphviz不可用，将DOT数据作为文本返回
            svg_content = None
        
        # 获取树的结构信息
        tree_depth = clf.tree_.max_depth
        leaf_count = clf.tree_.n_leaves
        
        # 生成树结构文本
        tree_structure = generate_tree_structure(clf, feature_names, target_encoder.classes_)
        
        return jsonify({
            'success': True,
            'train_accuracy': round(float(train_accuracy), 4),
            'test_accuracy': round(float(test_accuracy), 4),
            'tree_depth': int(tree_depth),
            'leaf_count': int(leaf_count),
            'train_time': round(float(train_time), 1),
            'tree_image': img_base64,
            'tree_structure': tree_structure,
            'graphviz_svg': svg_content,
            'dot_data': dot_data  # 添加DOT格式数据
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_tree_structure(clf, feature_names, class_names, node=0, depth=0):
    """生成树结构的文本表示"""
    tree = clf.tree_
    indent = "  " * depth
    
    if tree.feature[node] != -2:  # 不是叶节点
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        
        # 对于分类特征，显示具体的类别值
        if feature in label_encoders:
            # 找到最接近阈值的编码值
            encoder = label_encoders[feature]
            encoded_values = list(range(len(encoder.classes_)))
            threshold_int = int(round(threshold))
            if threshold_int < len(encoder.classes_):
                threshold_str = f"'{encoder.classes_[threshold_int]}'"
            else:
                threshold_str = str(threshold)
        else:
            threshold_str = f"{threshold:.3f}"
        
        structure = f"{indent}🌿 {feature} <= {threshold_str}\n"
        structure += generate_tree_structure(clf, feature_names, class_names, tree.children_left[node], depth + 1)
        structure += generate_tree_structure(clf, feature_names, class_names, tree.children_right[node], depth + 1)
        return structure
    else:  # 叶节点
        class_idx = np.argmax(tree.value[node])
        class_name = class_names[class_idx]
        samples = tree.n_node_samples[node]
        return f"{indent}🍃 预测: {class_name} (样本数: {samples})\n"

@app.route('/get_dataset_info', methods=['GET'])
def get_dataset_info():
    try:
        info = {
            'total_samples': len(watermelon_data),
            'features': feature_names,
            'target_classes': target_encoder.classes_.tolist(),
            'class_distribution': {
                cls: int(sum(watermelon_data['好瓜'] == cls)) 
                for cls in target_encoder.classes_
            }
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 启动时加载数据
    load_watermelon_data()
    print("西瓜数据集已加载")
    print(f"特征: {feature_names}")
    print(f"样本数: {len(watermelon_data)}")
    app.run(debug=True, port=5000)