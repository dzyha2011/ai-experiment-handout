#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN手写数字识别完整示例
作者: 实验讲义
功能: 使用卷积神经网络进行MNIST手写数字识别
"""

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def prepare_mnist_data():
    """准备MNIST数据集"""
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(np.int32)
    
    # 数据预处理
    X = X.reshape(-1, 1, 28, 28) / 255.0  # 归一化并添加通道维度
    y = torch.tensor(y, dtype=torch.long)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    """深层CNN模型（代码填空2的完整实现）"""
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 填空答案：32输入，64输出
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)  # 填空答案：64*3*3=576
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (16,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (32,7,7)
        x = self.pool(F.relu(self.conv3(x)))  # (64,3,3)
        x = x.view(-1, 64 * 3 * 3)  # 填空答案：展平操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=5):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 代码填空3的完整实现：学习率调度器
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm显示训练进度，设置合适的刷新频率
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, dynamic_ncols=True, mininterval=0.5)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条显示信息（每10个batch更新一次，减少IO开销）
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_acc = train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        train_loss = train_running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # 使用tqdm显示验证进度
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                           leave=False, dynamic_ncols=True, mininterval=0.5)
            for batch_idx, (inputs, labels) in enumerate(val_pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新验证进度条显示信息（每5个batch更新一次）
                if batch_idx % 5 == 0 or batch_idx == len(val_loader) - 1:
                    current_val_acc = val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_val_acc:.4f}'
                    })
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率（代码填空3答案）
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}\n")
    
    return history

def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    return test_acc, all_preds, all_labels

def plot_results(history, y_true, y_pred):
    """绘制结果"""
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制混淆矩阵
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.show()

def demonstrate_code_filling():
    """演示代码填空的正确答案"""
    print("\n=" * 50)
    print("代码填空答案演示")
    print("=" * 50)
    
    # 代码填空1答案
    print("\n代码填空1答案:")
    print("①处: X.reshape(-1, 1, 28, 28) / 255.0")
    print("②处: 10000")
    
    # 代码填空2答案
    print("\n代码填空2答案:")
    print("①处: 32 (输入通道数)")
    print("②处: 64 (输出通道数)")
    print("③处: 64 * 3 * 3 (特征数量)")
    print("④处: 64 * 3 * 3 (展平操作)")
    
    # 代码填空3答案
    print("\n代码填空3答案:")
    print("①处: StepLR")
    print("②处: 3")
    print("③处: 0.1")
    print("④处: scheduler.step()")

def main():
    """主函数"""
    print("=" * 60)
    print("CNN手写数字识别实验")
    print("=" * 60)
    
    # 演示代码填空答案
    demonstrate_code_filling()
    
    # 准备数据
    X_train, X_test, y_train, y_test = prepare_mnist_data()
    
    # 创建数据加载器
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 测试两种模型
    models_to_test = [
        ("SimpleCNN", SimpleCNN()),
        ("DeepCNN", DeepCNN())
    ]
    
    for model_name, model in models_to_test:
        print(f"\n{'='*20} 测试 {model_name} {'='*20}")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数数: {total_params:,}")
        
        # 训练模型
        print(f"\n开始训练 {model_name}...")
        history = train_model(model, train_loader, val_loader, num_epochs=5)
        
        # 评估模型
        print(f"\n在测试集上评估 {model_name}...")
        test_acc, y_pred, y_true = evaluate_model(model, test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), f'{model_name.lower()}_mnist_model.pth')
        print(f"模型已保存为 {model_name.lower()}_mnist_model.pth")
        
        # 绘制结果（仅为最后一个模型）
        if model_name == "DeepCNN":
            print("\n分类报告:")
            print(classification_report(y_true, y_pred))
            plot_results(history, y_true, y_pred)
    
    print("\n实验完成！")
    print("\n总结:")
    print("- 所有代码填空已正确实现")
    print("- 两种CNN模型都已训练完成")
    print("- 模型权重已保存")
    print("- 预期测试准确率: 98%+")

if __name__ == "__main__":
    main()