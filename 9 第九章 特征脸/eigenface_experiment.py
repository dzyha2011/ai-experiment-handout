import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2
import csv
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_images(path):
    """读取文件夹中的人脸图像，返回展平后的图像矩阵和标签"""
    # 获取所有图像路径
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('png')]
    
    # 读取第一张图片获取尺寸
    first_img = Image.open(image_paths[0]).convert('L')
    img_size = first_img.size[0] * first_img.size[1]
    
    # 初始化图像矩阵（行数=图像维度，列数=样本数）
    images = np.zeros((len(image_paths), img_size))
    labels = []
    
    for idx, img_path in enumerate(image_paths):
        # 读取图像并转为灰度图
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, 'uint8')
        # 展平为1维向量并存入矩阵
        img_flat = img_array.flatten()
        images[idx, :] = img_flat
        # 提取标签（文件名格式：personXX_XX.png，取XX部分）
        filename = os.path.basename(img_path)
        label = filename.split('_')[0][-2:]  # 提取person后的两位数字
        labels.append(label)
    
    return images.T, labels  # 返回形状为(D, N)的图像矩阵和标签列表

def eigen_train(trainset, K=10):
    """训练特征脸模型，返回top-K特征脸、平均脸、训练样本投影"""
    # 计算平均脸
    avg_img = np.mean(trainset, axis=1, keepdims=True)
    # 计算差异矩阵（每个样本减去平均脸）
    diff = trainset - avg_img
    # 计算协方差矩阵的特征值与特征向量（优化：使用X^T*X替代X*X^T）
    cov_matrix = (diff.T @ diff) / float(diff.shape[1])
    eig_vals, eig_vects = np.linalg.eig(cov_matrix)
    
    # 按特征值降序排序，取前K个特征向量
    sorted_idx = np.argsort(eig_vals)[::-1]
    top_eig_vects = eig_vects[:, sorted_idx[:K]]
    
    # 计算特征脸（差异矩阵×特征向量）
    eigenfaces = diff @ top_eig_vects
    
    # 特征脸归一化
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
    
    # 训练样本投影到特征脸空间
    trainset_proj = eigenfaces.T @ diff
    
    return eigenfaces, avg_img, trainset_proj, eig_vals[sorted_idx[:K]]

def eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, testset):
    """测试特征脸模型，返回预测标签"""
    # 测试样本减去平均脸
    diff_test = testset - avg_img
    # 投影到特征脸空间
    testset_proj = eigenfaces.T @ diff_test
    
    pred_labels = []
    for i in range(testset_proj.shape[1]):
        # 计算与所有训练样本的欧氏距离
        distances = np.linalg.norm(trainset_proj - testset_proj[:, i:i+1], axis=0)
        # 找最近邻
        min_idx = np.argmin(distances)
        pred_label = trainset_labels[min_idx]
        pred_labels.append(pred_label)
    
    return pred_labels

def visualize_eigenfaces(eigenfaces, avg_img, img_shape=(50, 50), save_path='eigenfaces_visualization.png'):
    """可视化平均脸和前几个特征脸"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 显示平均脸
    avg_face = avg_img.reshape(img_shape)
    axes[0, 0].imshow(avg_face, cmap='gray')
    axes[0, 0].set_title('平均脸')
    axes[0, 0].axis('off')
    
    # 显示前9个特征脸
    for i in range(9):
        row = i // 5 if i < 4 else (i + 1) // 5
        col = (i + 1) % 5 if i < 4 else (i + 1) % 5
        
        eigenface = eigenfaces[:, i].reshape(img_shape)
        # 归一化到0-255范围用于显示
        eigenface_norm = ((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255).astype(np.uint8)
        
        axes[row, col].imshow(eigenface_norm, cmap='gray')
        axes[row, col].set_title(f'特征脸 {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"特征脸可视化图片已保存为: {save_path}")

def plot_reconstruction(original_img, eigenfaces, avg_img, K_values, img_shape=(50, 50), save_path='reconstruction_comparison.png'):
    """展示不同K值下的图像重建效果"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原始图像
    axes[0, 0].imshow(original_img.reshape(img_shape), cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 不同K值的重建
    diff_img = original_img - avg_img
    
    for idx, K in enumerate(K_values):
        # 投影到K维特征脸空间
        projection = eigenfaces[:, :K].T @ diff_img
        # 重建图像
        reconstructed = avg_img + eigenfaces[:, :K] @ projection
        
        row = idx // 4
        col = (idx + 1) % 4
        
        axes[row, col].imshow(reconstructed.reshape(img_shape), cmap='gray')
        axes[row, col].set_title(f'K={K} 重建')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图像重建对比图已保存为: {save_path}")

def plot_accuracy_vs_k(K_values, accuracies, save_path='accuracy_vs_k.png'):
    """绘制识别准确率与K值的关系图"""
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('主成分数量 K')
    plt.ylabel('识别准确率')
    plt.title('特征脸方法：识别准确率与主成分数量K的关系')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_values)
    
    # 添加数值标签
    for i, (k, acc) in enumerate(zip(K_values, accuracies)):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"准确率-K值关系图已保存为: {save_path}")

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
    
    # 检查是否有训练数据
    if not os.path.exists('dataset') or len(os.listdir('dataset')) == 0:
        print("未找到训练数据，请先采集人脸样本")
        
        while True:
            choice = input("1. 采集新人脸样本\n2. 退出\n请选择: ")
            if choice == '1':
                name = input("请输入姓名: ")
                capture_face(name)
            elif choice == '2':
                return
            else:
                print("无效选择")
    
    # 加载训练数据
    print("加载训练数据...")
    all_images = []
    all_labels = []
    
    for person_dir in os.listdir('dataset'):
        person_path = os.path.join('dataset', person_dir)
        if os.path.isdir(person_path):
            images, labels = get_images(person_path)
            all_images.append(images)
            all_labels.extend([person_dir] * len(labels))
    
    if not all_images:
        print("未找到有效的训练数据")
        return
    
    trainset = np.hstack(all_images)
    trainset_labels = all_labels
    
    # 训练模型
    print("训练特征脸模型...")
    K = min(20, len(trainset_labels) - 1)  # K不能超过样本数-1
    eigenfaces, avg_img, trainset_proj, _ = eigen_train(trainset, K)
    
    print(f"模型训练完成，使用 {K} 个主成分")
    print("开始实时考勤识别...")
    
    # 实时识别
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (50, 50))
            face_vector = face_roi.flatten().reshape(-1, 1)
            
            # 识别
            pred_labels = eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, face_vector)
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
        
        cv2.imshow('考勤系统', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # 按c键采集新样本
            name = input("请输入新人员姓名: ")
            cap.release()
            cv2.destroyAllWindows()
            capture_face(name)
            cap = cv2.VideoCapture(0)
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """主实验函数"""
    print("=== 特征脸方法实验 ===")
    
    # 检查数据集是否存在
    if not os.path.exists('./yaleBfaces/subset0'):
        print("未找到YaleB人脸数据集，请确保数据集路径正确")
        return
    
    # 1. 加载数据
    print("1. 加载训练和测试数据...")
    trainset, trainset_labels = get_images('./yaleBfaces/subset0')  # 训练集：subset0
    testset, testset_labels = get_images('./yaleBfaces/subset1')    # 测试集：subset1
    
    print(f"训练集: {trainset.shape[1]} 张图片")
    print(f"测试集: {testset.shape[1]} 张图片")
    print(f"图像尺寸: {int(np.sqrt(trainset.shape[0]))}x{int(np.sqrt(trainset.shape[0]))}")
    
    # 2. 训练模型并可视化
    print("\n2. 训练特征脸模型...")
    K = 20
    eigenfaces, avg_img, trainset_proj, eigenvalues = eigen_train(trainset, K)
    
    # 可视化特征脸
    img_size = int(np.sqrt(trainset.shape[0]))
    visualize_eigenfaces(eigenfaces, avg_img, (img_size, img_size))
    
    # 3. 测试不同K值的性能
    print("\n3. 测试不同K值的识别性能...")
    K_values = [5, 10, 15, 20, 25, 30]
    accuracies = []
    
    for k in K_values:
        eigenfaces_k, avg_img_k, trainset_proj_k, _ = eigen_train(trainset, k)
        pred_labels = eigen_test(trainset_proj_k, avg_img_k, eigenfaces_k, trainset_labels, testset)
        
        correct = sum(p == t for p, t in zip(pred_labels, testset_labels))
        accuracy = correct / len(testset_labels)
        accuracies.append(accuracy)
        print(f"K={k:2d}: 准确率 = {accuracy:.3f}")
    
    # 绘制准确率-K值关系图
    plot_accuracy_vs_k(K_values, accuracies)
    
    # 4. 图像重建演示
    print("\n4. 图像重建演示...")
    # 选择一张测试图像进行重建
    test_img = testset[:, 0:1]  # 第一张测试图像
    K_recon = [1, 5, 10, 20, 30, 50, 100]
    plot_reconstruction(test_img, eigenfaces, avg_img, K_recon, (img_size, img_size))
    
    # 5. 测试不同光照条件的性能
    print("\n5. 测试不同光照条件下的性能...")
    subsets = ['subset1', 'subset2', 'subset3', 'subset4']
    subset_names = ['弱阴影', '中等阴影', '强阴影', '极端阴影']
    
    print("光照条件\t准确率")
    print("-" * 20)
    for subset, name in zip(subsets, subset_names):
        subset_path = f'./yaleBfaces/{subset}'
        if os.path.exists(subset_path):
            testset_subset, testset_labels_subset = get_images(subset_path)
            pred_labels_subset = eigen_test(trainset_proj, avg_img, eigenfaces, trainset_labels, testset_subset)
            
            correct = sum(p == t for p, t in zip(pred_labels_subset, testset_labels_subset))
            accuracy = correct / len(testset_labels_subset)
            print(f"{name}\t\t{accuracy:.3f}")
    
    print("\n实验完成！生成的图片文件:")
    print("- eigenfaces_visualization.png: 特征脸可视化")
    print("- reconstruction_comparison.png: 图像重建对比")
    print("- accuracy_vs_k.png: 准确率-K值关系图")

if __name__ == "__main__":
    while True:
        print("\n=== 特征脸实验系统 ===")
        print("1. 运行完整实验")
        print("2. 启动考勤系统")
        print("3. 采集人脸样本")
        print("4. 退出")
        
        choice = input("请选择功能 (1-4): ")
        
        if choice == '1':
            main()
        elif choice == '2':
            attendance_system()
        elif choice == '3':
            name = input("请输入姓名: ")
            capture_face(name)
        elif choice == '4':
            print("再见！")
            break
        else:
            print("无效选择，请重新输入")