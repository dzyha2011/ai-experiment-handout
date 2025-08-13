# 循环神经网络实验 - PyTorch版本

本实验讲义已完全转换为PyTorch深度学习框架，包含IMDB情感分类和股票价格预测两个完整实例。

## 环境要求

- Python 3.7+
- PyTorch 2.0+
- CUDA支持（可选，用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install torch numpy matplotlib pandas scikit-learn
```

## 文件说明

- `循环神经网络实验讲义-2c2974669f.md` - 完整的实验讲义（PyTorch版本）
- `test_rnn_pytorch.py` - 完整的测试脚本，验证所有代码功能
- `requirements.txt` - Python依赖包列表
- `README.md` - 本说明文件

## 快速开始

1. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行测试脚本验证环境：
   ```bash
   python test_rnn_pytorch.py
   ```

3. 查看实验讲义，按照步骤进行学习

## 实验内容

### 1. IMDB情感分类
- 使用SimpleRNN进行文本情感分类
- 包含完整的数据预处理、模型构建、训练和评估
- 支持GPU加速训练

### 2. 股票价格预测
- 使用LSTM进行时间序列预测
- 包含数据生成、序列构建、模型训练和结果可视化
- 提供多种评估指标（RMSE、MAE、MAPE）

## 主要特性

✅ **完全PyTorch实现** - 所有代码都使用PyTorch框架
✅ **GPU支持** - 自动检测并使用CUDA加速
✅ **完整可运行** - 所有代码都经过测试验证
✅ **详细注释** - 代码包含详细的中文注释
✅ **模块化设计** - 代码结构清晰，易于理解和修改
✅ **可视化支持** - 包含训练过程和结果的可视化

## 代码结构

```
RNN/
├── 循环神经网络实验讲义-2c2974669f.md  # 主要实验讲义
├── test_rnn_pytorch.py                    # 测试脚本
├── requirements.txt                       # 依赖包
└── README.md                             # 说明文件
```

## 运行结果示例

测试脚本运行后会显示：
- IMDB情感分类的训练过程和测试准确率
- 股票价格预测的训练损失和评估指标
- 股票价格预测结果的可视化图表

## 注意事项

1. 如果没有GPU，代码会自动使用CPU运行
2. 实验讲义中的数据集为模拟数据，实际使用时可替换为真实数据
3. 可以根据需要调整模型参数和超参数
4. 建议在Jupyter Notebook中运行以获得更好的交互体验

## 技术支持

如果遇到问题，请检查：
1. Python版本是否符合要求
2. 依赖包是否正确安装
3. CUDA版本是否与PyTorch兼容（如使用GPU）

## 更新日志

- 2024年：完全转换为PyTorch框架
- 添加了完整的测试脚本
- 优化了代码结构和注释
- 增加了GPU支持和错误处理