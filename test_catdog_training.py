#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试猫狗分类训练代码 - 验证进度条是否正常工作
这是一个简化的测试版本，用于验证训练流程不会卡住
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

# 创建模拟数据集
class MockCatDogDataset(Dataset):
    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # 模拟标签：0=猫，1=狗
        self.labels = torch.randint(0, 2, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 创建随机图像数据 (3, 224, 224)
        image = torch.randn(3, 224, 224)
        label = self.labels[idx]
        
        if self.transform:
            # 将tensor转换为PIL图像再应用变换
            image = transforms.ToPILImage()(image)
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
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def test_training_progress():
    """测试训练进度条是否正常工作"""
    print("开始测试猫狗分类训练进度条...")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建模拟数据集
    train_dataset = MockCatDogDataset(num_samples=200, transform=transform)
    val_dataset = MockCatDogDataset(num_samples=50, transform=transform)
    
    # Windows系统设置num_workers=0避免卡住
    num_workers = 0 if os.name == 'nt' else 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = SimpleResNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 测试训练循环（只训练1个epoch）
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    print("\n开始训练测试...")
    # 使用优化后的tqdm设置
    train_pbar = tqdm(train_loader, desc='测试训练进度', 
                     leave=False, dynamic_ncols=True, mininterval=0.5)
    
    for batch_idx, (inputs, labels) in enumerate(train_pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # 每5个batch更新一次进度条
        if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
            current_acc = train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}'
            })
    
    print(f"训练完成 - 损失: {train_loss/len(train_loader):.4f}, 准确率: {train_correct/train_total:.4f}")
    
    # 测试验证循环
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    print("\n开始验证测试...")
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc='测试验证进度', 
                       leave=False, dynamic_ncols=True, mininterval=0.5)
        
        for batch_idx, (inputs, labels) in enumerate(val_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # 每3个batch更新一次进度条
            if batch_idx % 3 == 0 or batch_idx == len(val_loader) - 1:
                current_val_acc = val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.4f}'
                })
    
    print(f"验证完成 - 损失: {val_loss/len(val_loader):.4f}, 准确率: {val_correct/val_total:.4f}")
    print("\n✅ 进度条测试成功！训练和验证过程都能正常显示进度。")

if __name__ == '__main__':
    test_training_progress()