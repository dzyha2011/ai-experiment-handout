#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch RNN实验测试脚本
测试IMDB情感分类和股票价格预测的完整功能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== IMDB情感分类测试 ====================

class SimpleRNNModel(nn.Module):
    """简单RNN模型用于情感分类"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        last_output = rnn_out[:, -1, :]
        output = self.fc(last_output)
        output = self.sigmoid(output)
        return output

def test_imdb_classification():
    """测试IMDB情感分类功能"""
    print("\n=== 测试IMDB情感分类 ===")
    
    # 模拟IMDB数据集
    vocab_size = 1000
    max_len = 100
    num_samples = 200
    
    x_train = [np.random.randint(1, vocab_size, size=np.random.randint(20, max_len)) 
               for _ in range(num_samples)]
    y_train = np.random.randint(0, 2, size=num_samples)
    
    x_test = [np.random.randint(1, vocab_size, size=np.random.randint(20, max_len)) 
              for _ in range(50)]
    y_test = np.random.randint(0, 2, size=50)
    
    # 数据预处理
    def pad_sequences_pytorch(sequences, maxlen=100, value=0):
        padded_sequences = []
        for seq in sequences:
            seq = list(seq)
            if len(seq) > maxlen:
                seq = seq[-maxlen:]
            if len(seq) < maxlen:
                seq = [value] * (maxlen - len(seq)) + seq
            padded_sequences.append(seq)
        return np.array(padded_sequences)
    
    x_train = pad_sequences_pytorch(x_train, maxlen=max_len)
    x_test = pad_sequences_pytorch(x_test, maxlen=max_len)
    
    # 转换为张量
    x_train = torch.LongTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_test = torch.LongTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 构建模型
    model = SimpleRNNModel(vocab_size, 32, 32, 1)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 5
    print("开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        output = model(x_test).squeeze()
        predicted = (output > 0.5).float()
        test_acc = (predicted == y_test).float().mean().item()
    
    print(f"测试集准确率：{test_acc:.4f}")
    print("IMDB情感分类测试完成！")
    
    return model

# ==================== 股票价格预测测试 ====================

class StockLSTM(nn.Module):
    """LSTM模型用于股票价格预测"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        
        return out

def generate_stock_data(n_days=300):
    """生成模拟股票价格数据"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 生成带趋势的随机游走
    price = 100  # 初始价格
    prices = [price]
    
    for i in range(1, n_days):
        # 添加趋势和随机波动
        trend = 0.001  # 轻微上升趋势
        volatility = 0.02
        change = np.random.normal(trend, volatility)
        price = price * (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    df.set_index('Date', inplace=True)
    
    return df

def create_sequences(data, seq_length):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def test_stock_prediction():
    """测试股票价格预测功能"""
    print("\n=== 测试股票价格预测 ===")
    
    # 生成数据
    df = generate_stock_data(300)
    print(f"数据形状: {df.shape}")
    
    # 数据预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])
    
    # 创建序列
    seq_length = 30
    X, y = create_sequences(df_scaled, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 构建模型
    model = StockLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 10
    print("开始训练...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    # 预测
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        predictions = model(X_test).cpu().numpy()
    
    # 反归一化
    predictions = scaler.inverse_transform(predictions)
    y_test_original = scaler.inverse_transform(y_test.numpy())
    
    # 计算评估指标
    rmse = np.sqrt(np.mean((predictions - y_test_original) ** 2))
    mae = np.mean(np.abs(predictions - y_test_original))
    
    print(f"测试集RMSE：{rmse:.4f}")
    print(f"测试集MAE：{mae:.4f}")
    
    # 简单可视化（前20个预测点）
    plt.figure(figsize=(10, 6))
    n_show = min(20, len(y_test_original))
    plt.plot(range(n_show), y_test_original[:n_show], 'b-o', label='Actual Price', markersize=4)
    plt.plot(range(n_show), predictions[:n_show], 'r-s', label='Predicted Price', markersize=4)
    plt.title('Stock Price Prediction (First 20 Points)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("股票价格预测测试完成！")
    
    return model

# ==================== 主测试函数 ====================

def main():
    """主测试函数"""
    print("开始PyTorch RNN实验测试...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {device}")
    
    try:
        # 测试IMDB情感分类
        rnn_model = test_imdb_classification()
        
        # 测试股票价格预测
        lstm_model = test_stock_prediction()
        
        print("\n=== 所有测试完成 ===")
        print("✅ IMDB情感分类测试通过")
        print("✅ 股票价格预测测试通过")
        print("✅ PyTorch RNN实验代码运行正常")
        
        # 保存模型
        torch.save(rnn_model.state_dict(), 'test_rnn_model.pth')
        torch.save(lstm_model.state_dict(), 'test_lstm_model.pth')
        print("✅ 模型已保存")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()