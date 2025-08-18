from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# 定义CNN模型结构（与训练时保持一致）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 加载模型 - 强制使用CPU避免CUDA库问题
device = torch.device('cpu')

# 初始化模型
simple_model = SimpleCNN(num_classes=10)
deep_model = DeepCNN(num_classes=10)

# 加载权重并移动到正确设备
try:
    if os.path.exists('simplecnn_mnist_model.pth'):
        simple_model.load_state_dict(torch.load('simplecnn_mnist_model.pth', map_location=device))
        simple_model = simple_model.to(device)
        simple_model.eval()
        print("简单CNN模型加载成功")
except Exception as e:
    print(f"简单CNN模型加载失败: {e}")

try:
    if os.path.exists('deepcnn_mnist_model.pth'):
        deep_model.load_state_dict(torch.load('deepcnn_mnist_model.pth', map_location=device))
        deep_model = deep_model.to(device)
        deep_model.eval()
        print("深度CNN模型加载成功")
except Exception as e:
    print(f"深度CNN模型加载失败: {e}")

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((28, 28)),  # 调整为28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的标准化参数
])

# MNIST数字标签
mnist_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取上传的图片
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 获取模型类型
        model_type = request.form.get('model_type', 'simple')
        
        # 读取图片
        image = Image.open(file.stream)
        
        # 预处理图片
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 选择模型进行预测
        with torch.no_grad():
            if model_type == 'deep':
                outputs = deep_model(input_tensor)
                model_name = "深度CNN模型"
            else:
                outputs = simple_model(input_tensor)
                model_name = "简单CNN模型"
            
            # 获取预测结果
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 获取所有类别的概率
            all_probs = probabilities[0].cpu().numpy()
            
            result = {
                'success': True,
                'model_name': model_name,
                'predicted_class': mnist_labels[predicted.item()],
                'confidence': float(confidence.item()),
                'all_probabilities': {
                    mnist_labels[i]: float(prob) for i, prob in enumerate(all_probs)
                }
            }
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'simple_cnn': os.path.exists('simplecnn_mnist_model.pth'),
            'deep_cnn': os.path.exists('deepcnn_mnist_model.pth')
        }
    })

if __name__ == '__main__':
    print("🚀 CNN图像分类服务启动中...")
    print(f"📱 设备: {device}")
    print("🔗 API端点:")
    print("   - POST /predict - 图片分类预测")
    print("   - GET /health - 健康检查")
    print("\n💡 使用说明:")
    print("   1. 上传28x28像素的手写数字图片")
    print("   2. 选择模型类型 (simple/deep)")
    print("   3. 获取分类结果和置信度")
    
    app.run(host='0.0.0.0', port=5000, debug=True)