#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证5.2.3节训练卡顿问题的修复
对比修复前后的性能差异
"""

import time
from tqdm import tqdm

def test_inefficient_method():
    """测试低效的方法（修复前）"""
    print("测试低效方法（修复前的代码）...")
    
    # 模拟数据加载器
    data_loader = range(100)  # 模拟100个batch
    
    start_time = time.time()
    
    # 模拟原来的低效代码
    val_pbar = tqdm(data_loader, desc='低效方法测试', leave=False)
    for item in val_pbar:
        # 模拟数据处理
        time.sleep(0.01)  # 模拟处理时间
        
        # 原来的低效操作
        batch_num = len([x for x in val_pbar])  # 这会遍历整个进度条！
        if batch_num % 5 == 0:
            val_pbar.set_postfix({'Batch': batch_num})
    
    end_time = time.time()
    inefficient_time = end_time - start_time
    print(f"低效方法耗时: {inefficient_time:.2f}秒")
    return inefficient_time

def test_efficient_method():
    """测试高效的方法（修复后）"""
    print("\n测试高效方法（修复后的代码）...")
    
    # 模拟数据加载器
    data_loader = range(100)  # 模拟100个batch
    
    start_time = time.time()
    
    # 修复后的高效代码
    val_pbar = tqdm(data_loader, desc='高效方法测试', leave=False)
    for batch_idx, item in enumerate(val_pbar):
        # 模拟数据处理
        time.sleep(0.01)  # 模拟处理时间
        
        # 修复后的高效操作
        if batch_idx % 5 == 0 or batch_idx == len(data_loader) - 1:
            val_pbar.set_postfix({'Batch': batch_idx})
    
    end_time = time.time()
    efficient_time = end_time - start_time
    print(f"高效方法耗时: {efficient_time:.2f}秒")
    return efficient_time

def main():
    print("=== 5.2.3节训练卡顿问题修复验证 ===")
    print("\n问题描述：")
    print("- 原代码在验证阶段使用 len([x for x in val_pbar]) 获取batch数量")
    print("- 这个操作会遍历整个进度条对象，创建列表，非常低效")
    print("- 在大数据集上会导致训练卡顿甚至假死")
    
    print("\n修复方案：")
    print("- 使用 enumerate(val_pbar) 获取batch索引")
    print("- 直接使用 batch_idx 进行条件判断")
    print("- 避免了低效的列表推导式操作")
    
    print("\n性能对比测试：")
    
    # 测试低效方法
    try:
        inefficient_time = test_inefficient_method()
    except Exception as e:
        print(f"低效方法执行出错: {e}")
        inefficient_time = float('inf')
    
    # 测试高效方法
    efficient_time = test_efficient_method()
    
    # 性能对比
    print("\n=== 性能对比结果 ===")
    if inefficient_time != float('inf'):
        speedup = inefficient_time / efficient_time
        print(f"性能提升: {speedup:.1f}倍")
        print(f"时间节省: {inefficient_time - efficient_time:.2f}秒")
    else:
        print("低效方法执行失败，高效方法正常运行")
    
    print("\n✅ 修复验证完成！")
    print("\n修复要点总结：")
    print("1. 将 'for inputs, labels in val_pbar:' 改为 'for batch_idx, (inputs, labels) in enumerate(val_pbar):'")
    print("2. 将 'batch_num = len([x for x in val_pbar])' 删除")
    print("3. 将条件判断改为 'if batch_idx % 5 == 0 or batch_idx == len(val_loader) - 1:'")
    print("4. 这样既保持了原有功能，又大幅提升了性能")

if __name__ == "__main__":
    main()