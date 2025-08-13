import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

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
    main()