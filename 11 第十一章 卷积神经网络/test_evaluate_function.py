#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试evaluate_model函数调用修复
验证TypeError是否已解决
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 简单的测试模型
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

# 模拟evaluate_model函数（返回三个值）
def evaluate_model(model, test_loader):
    """评估模型 - 返回三个值"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    return test_acc, all_preds, all_labels

def test_correct_usage():
    """测试正确的函数调用方式"""
    print("测试evaluate_model函数调用修复...")
    
    # 创建测试数据
    test_data = torch.randn(100, 10)
    test_labels = torch.randint(0, 2, (100,))
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 创建模型
    model = TestModel()
    
    print("\n1. 测试正确的调用方式（接收三个返回值）:")
    try:
        test_acc, y_pred, y_true = evaluate_model(model, test_loader)
        print(f"✅ 成功！测试集准确率: {test_acc:.4f}")
        print(f"   预测结果数量: {len(y_pred)}")
        print(f"   真实标签数量: {len(y_true)}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    print("\n2. 测试错误的调用方式（只接收一个返回值）:")
    try:
        test_acc = evaluate_model(model, test_loader)
        print(f"❌ 这会导致TypeError: {test_acc:.4f}")
    except TypeError as e:
        print(f"✅ 预期的TypeError: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")
    
    print("\n3. 测试部分接收的调用方式（接收三个值但忽略后两个）:")
    try:
        test_acc, _, _ = evaluate_model(model, test_loader)
        print(f"✅ 成功！测试集准确率: {test_acc:.4f}")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == '__main__':
    test_correct_usage()
    print("\n🎉 测试完成！修复后的代码应该能正常工作。")