import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleCNN()

# 随机生成一个输入张量（例如MNIST图像 1x28x28）
x = torch.randn(1, 1, 28, 28)

# 生成网络结构图
y = model(x)
dot = make_dot(y, params=dict(list(model.named_parameters())))
dot.render("simple_cnn_architecture", format="png")