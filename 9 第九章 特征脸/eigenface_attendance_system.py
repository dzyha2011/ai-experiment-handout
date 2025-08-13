#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征脸人脸识别考勤系统
基于主成分分析(PCA)的人脸识别与自动考勤

作者：AI实验课程组
功能：
1. 人脸数据采集
2. 特征脸模型训练
3. 实时人脸识别
4. 自动考勤记录
5. 实验结果可视化

使用说明：
1. 确保安装依赖：pip install numpy matplotlib pillow opencv-python
2. 运行程序：python eigenface_attendance_system.py
3. 按菜单提示操作
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
import csv
from datetime import datetime

# ==================== 核心算法模块 ====================

def get_images(path):
    """读取文件夹中的人脸图像，返回展平后的图像矩阵和标签"""
    if not os.path.exists(path):
        print(f"警告：路径 {path} 不存在")
        return np.array([]).reshape(2500, 0), []
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    if not image_paths:
        print(f"警告：路径 {path} 中未找到PNG图像文件")
        return np.array([]).reshape(2500, 0), []
    
    images = np.zeros((len(image_paths), 2500))
    labels = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, 'uint8')
            img_flat = img_array.flatten()
            images[idx, :] = img_flat
            
            # 从文件名提取标签（支持多种格式）
            filename = os.path.basename(img_path)
            if 'person' in filename:
                label = filename.split('_')[0]  # person01 -> person01
            else:
                label = filename.split('_')[0]  # name_01.png -> name
            labels.append(label)
        except Exception as e:
            print(f"读取图像 {img_path} 失败: {e}")
    
    return images.T, labels

def eigen_train(trainset, K=10):
    """训练特征脸模型，返回top-K特征脸、平均脸、训练样本投影"""
    if trainset.shape[1] == 0:
        print("错误：训练集为空")
        return None, None, None
    
    # 计算平均脸
    avg_img = np.mean(trainset, axis=1, keepdims=True)
    # 计算差异矩阵
    diff = trainset - avg_img
    # 计算协方差矩阵（使用优化方法）
    cov_matrix = (diff.T @ diff) / float(diff.shape[1])
    eig_vals, eig_vects = np.linalg.eig(cov_matrix)
    
    # 特征向量归一化
    eig_vects = eig_vects / np.linalg.norm(eig_vects, axis=0)
    # 按特征值降序排序
    sorted_idx = np.argsort(eig_vals)[::-1]
    top_eig_vects = eig_vects[:, sorted_idx[:K]]
    
    # 计算特征脸
    eigenfaces = diff @ top_eig_vects
    # 训练样本投影
    trainset_proj = eigenfaces.T @ diff
    
    return eigenfaces, avg_img, trainset_proj

def eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, testset):
    """测试特征脸模型，返回预测标签"""
    if testset.shape[1] == 0:
        return []
    
    diff_test = testset - avg_img
    testset_proj = eigenfaces.T @ diff_test
    
    pred_labels = []
    for i in range(testset_proj.shape[1]):
        distances = np.linalg.norm(trainset_proj - testset_proj[:, i:i+1], axis=0)
        min_idx = np.argmin(distances)
        pred_labels.append(trainset_labels[min_idx])
    
    return pred_labels

# ==================== 数据采集模块 ====================

def capture_face(name, num_samples=5):
    """调用摄像头采集人脸样本，保存至dataset/name/"""
    dataset_path = f'dataset/{name}'
    os.makedirs(dataset_path, exist_ok=True)
    
    cap = cv2.VideoCapture(0)  # 打开摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    
    print(f"开始采集 {name} 的人脸样本，需要 {num_samples} 张照片")
    print("按空格键拍照，按q键退出")
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
            
        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 绘制人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} - {count}/{num_samples}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('人脸采集', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:  # 按空格键拍照
            # 取第一个检测到的人脸
            x, y, w, h = faces[0]
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            
            # 保存图像
            filename = f'{dataset_path}/{name}_{count:02d}.png'
            cv2.imwrite(filename, face_roi)
            print(f"已保存: {filename}")
            count += 1
            
        elif key == ord('q'):  # 按q键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"人脸采集完成，共采集 {count} 张照片")

# ==================== 考勤记录模块 ====================

def record_attendance(pred_label, confidence=0.0, attendance_file='attendance.csv'):
    """记录考勤结果（姓名、时间、状态、置信度）"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 检查文件是否存在，不存在则创建并写入表头
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['姓名', '时间', '状态', '置信度'])
    
    # 读取今日已有记录，避免重复
    today = datetime.now().strftime('%Y-%m-%d')
    existing_records = []
    
    try:
        with open(attendance_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 2 and row[1].startswith(today):
                    existing_records.append(row[0])
    except FileNotFoundError:
        pass
    
    # 记录考勤
    if pred_label not in existing_records:
        with open(attendance_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([pred_label, now, 'Present', f'{confidence:.3f}'])
        print(f"考勤成功：{pred_label} {now} (置信度: {confidence:.3f})")
        return True
    else:
        print(f"今日已考勤：{pred_label}")
        return False

def attendance_system():
    """基于特征脸的考勤系统主函数"""
    print("=== 特征脸考勤系统 ===")
    
    # 检查数据集是否存在
    dataset_path = './dataset'
    if not os.path.exists(dataset_path):
        print("数据集不存在，请先采集人脸数据")
        return
    
    # 加载训练数据
    print("正在加载训练数据...")
    all_images = []
    all_labels = []
    
    # 遍历dataset文件夹中的所有子文件夹
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            images, labels = get_images(person_path)
            if images.shape[1] > 0:
                all_images.append(images)
                all_labels.extend([person_name] * len(labels))
                print(f"加载 {person_name}: {images.shape[1]} 张图片")
    
    if not all_images:
        print("未找到训练数据，请先采集人脸数据")
        return
    
    # 合并所有图像数据
    trainset = np.hstack(all_images)
    trainset_labels = all_labels
    print(f"训练集总计: {trainset.shape[1]} 张图片，{len(set(trainset_labels))} 个人")
    
    # 训练特征脸模型
    print("正在训练特征脸模型...")
    K = min(20, trainset.shape[1] - 1)  # 确保K不超过样本数-1
    eigenfaces, avg_img, trainset_proj = eigen_train(trainset, K)
    
    if eigenfaces is None:
        print("模型训练失败")
        return
    
    print(f"模型训练完成，使用 {K} 个主成分")
    
    # 实时识别
    print("启动摄像头进行实时识别...")
    print("按 'q' 键退出系统")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            face_vector = face_roi.flatten().reshape(-1, 1)
            
            # 识别
            pred_labels = eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, face_vector)
            if pred_labels:
                pred_label = pred_labels[0]
                
                # 计算置信度（距离的倒数）
                diff_test = face_vector - avg_img
                test_proj = eigenfaces.T @ diff_test
                distances = np.linalg.norm(trainset_proj - test_proj, axis=0)
                min_distance = np.min(distances)
                confidence = 1.0 / (1.0 + min_distance / 1000)  # 归一化置信度
                
                # 绘制结果
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f'{pred_label} ({confidence:.2f})', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 自动考勤（置信度足够高时）
                if confidence > 0.6:
                    record_attendance(pred_label, confidence)
            else:
                # 未识别出人脸
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('考勤系统', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("考勤系统已关闭")

# ==================== 实验演示模块 ====================

def run_basic_experiment():
    """运行基础特征脸实验（使用YaleB数据集）"""
    print("=== 基础特征脸实验 ===")
    
    # 检查YaleB数据集
    if not os.path.exists('./yaleBfaces/subset0'):
        print("错误：未找到YaleB数据集，请确保yaleBfaces文件夹存在")
        return
    
    # 加载训练和测试数据
    print("正在加载YaleB数据集...")
    trainset, trainset_labels = get_images('./yaleBfaces/subset0')
    testset, testset_labels = get_images('./yaleBfaces/subset1')
    
    if trainset.shape[1] == 0 or testset.shape[1] == 0:
        print("数据加载失败")
        return
    
    print(f"训练集: {trainset.shape[1]} 张图片")
    print(f"测试集: {testset.shape[1]} 张图片")
    
    # 测试不同K值的性能
    K_values = [5, 10, 15, 20, 25, 30]
    accuracies = []
    
    for K in K_values:
        print(f"\n测试 K={K}...")
        eigenfaces, avg_img, trainset_proj = eigen_train(trainset, K)
        
        if eigenfaces is None:
            continue
            
        pred_labels = eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, testset)
        
        # 计算准确率
        correct = sum(p == t for p, t in zip(pred_labels, testset_labels))
        accuracy = correct / len(testset_labels)
        accuracies.append(accuracy)
        
        print(f"K={K}, 准确率: {accuracy:.3f} ({correct}/{len(testset_labels)})")
    
    # 显示最佳结果
    if accuracies:
        best_idx = np.argmax(accuracies)
        best_K = K_values[best_idx]
        best_accuracy = accuracies[best_idx]
        print(f"\n最佳结果: K={best_K}, 准确率={best_accuracy:.3f}")
    
    print("基础实验完成")

def visualize_eigenfaces():
    """可视化特征脸"""
    print("=== 特征脸可视化 ===")
    
    # 检查数据集
    if os.path.exists('./yaleBfaces/subset0'):
        trainset, _ = get_images('./yaleBfaces/subset0')
    elif os.path.exists('./dataset'):
        # 使用自定义数据集
        all_images = []
        for person_name in os.listdir('./dataset'):
            person_path = os.path.join('./dataset', person_name)
            if os.path.isdir(person_path):
                images, _ = get_images(person_path)
                if images.shape[1] > 0:
                    all_images.append(images)
        if all_images:
            trainset = np.hstack(all_images)
        else:
            print("未找到可用的数据集")
            return
    else:
        print("未找到数据集")
        return
    
    if trainset.shape[1] == 0:
        print("数据集为空")
        return
    
    # 训练模型
    K = min(9, trainset.shape[1] - 1)
    eigenfaces, avg_img, _ = eigen_train(trainset, K)
    
    if eigenfaces is None:
        return
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 显示平均脸
    plt.subplot(3, 4, 1)
    avg_face = avg_img.reshape(50, 50)
    plt.imshow(avg_face, cmap='gray')
    plt.title('平均脸')
    plt.axis('off')
    
    # 显示前9个特征脸
    for i in range(min(9, K)):
        plt.subplot(3, 4, i + 2)
        eigenface = eigenfaces[:, i].reshape(50, 50)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'特征脸 {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('eigenfaces_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("特征脸可视化完成，已保存为 eigenfaces_visualization.png")

# ==================== 主函数 ====================

def main():
    """主函数 - 提供交互式菜单"""
    print("\n" + "="*50)
    print("     特征脸人脸识别系统")
    print("     基于主成分分析(PCA)的人脸识别")
    print("="*50)
    
    while True:
        print("\n请选择功能:")
        print("1. 采集人脸数据")
        print("2. 运行基础实验（YaleB数据集）")
        print("3. 可视化特征脸")
        print("4. 启动考勤系统")
        print("5. 查看考勤记录")
        print("0. 退出程序")
        
        try:
            choice = input("\n请输入选择 (0-5): ").strip()
            
            if choice == '0':
                print("程序退出，再见！")
                break
            elif choice == '1':
                name = input("请输入姓名: ").strip()
                if name:
                    num_samples = input("请输入采集样本数量 (默认5): ").strip()
                    num_samples = int(num_samples) if num_samples.isdigit() else 5
                    capture_face(name, num_samples)
                else:
                    print("姓名不能为空")
            elif choice == '2':
                run_basic_experiment()
            elif choice == '3':
                visualize_eigenfaces()
            elif choice == '4':
                attendance_system()
            elif choice == '5':
                # 查看考勤记录
                if os.path.exists('attendance.csv'):
                    print("\n=== 考勤记录 ===")
                    with open('attendance.csv', 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content)
                else:
                    print("暂无考勤记录")
            else:
                print("无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            print("请重新选择")

if __name__ == "__main__":
    main()