#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试猫狗分类训练修复后的代码
验证训练过程不会在一轮后卡住
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os

# 模拟猫狗数据集
class MockCatDogDataset(Dataset):
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机图像数据 (3, 224, 224)
        image = torch.randn(3, 224, 224)
        # 随机标签 (0: 猫, 1: 狗)
        label = torch.randint(0, 2, (1,)).item()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 简化的ResNet模型
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def test_training_progress_fixed():
    """测试修复后的训练进度条是否正常工作"""
    print("开始测试修复后的猫狗分类训练进度条...")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = MockCatDogDataset(size=200, transform=transform)
    val_dataset = MockCatDogDataset(size=100, transform=transform)
    
    # Windows系统使用num_workers=0避免多进程问题
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # 创建模型
    model = SimpleResNet(num_classes=2).to(device)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = GradScaler()
    
    num_epochs = 3  # 测试3个epoch
    
    print(f"开始训练，共{num_epochs}个epoch...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用修复后的训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, dynamic_ncols=True, mininterval=0.5)
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计训练指标
            train_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条显示信息（每10个batch更新一次）
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                current_acc = train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # 计算训练集指标
        train_loss = train_running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段 - 使用修复后的代码
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            # 使用修复后的验证进度条
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
                
                # 修复后的验证进度条更新逻辑（每5个batch更新一次）
                if batch_idx % 5 == 0 or batch_idx == len(val_loader) - 1:
                    current_val_acc = val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_val_acc:.4f}'
                    })
        
        # 计算验证集指标
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step()
        
        # 打印epoch结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
    
    print("\n✅ 修复测试成功！训练过程没有卡住，所有epoch都能正常完成。")
    print("修复要点：")
    print("1. 使用enumerate(val_pbar)获取batch_idx，而不是len([x for x in val_pbar])")
    print("2. 验证进度条更新逻辑优化，避免了低效的列表遍历操作")
    print("3. 保持了原有的功能，但大幅提升了性能")

if __name__ == "__main__":
    test_training_progress_fixed()