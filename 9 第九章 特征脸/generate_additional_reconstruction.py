import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_images(path):
    """读取文件夹中的人脸图像，返回展平后的图像矩阵和标签"""
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('png')]
    first_img = Image.open(image_paths[0]).convert('L')
    img_size = first_img.size[0] * first_img.size[1]
    
    images = np.zeros((len(image_paths), img_size))
    labels = []
    
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, 'uint8')
        img_flat = img_array.flatten()
        images[idx, :] = img_flat
        filename = os.path.basename(img_path)
        label = filename.split('_')[0][-2:]
        labels.append(label)
    
    return images.T, labels

def eigen_train(trainset, K=10):
    """训练特征脸模型，返回top-K特征脸、平均脸、训练样本投影"""
    avg_img = np.mean(trainset, axis=1, keepdims=True)
    diff = trainset - avg_img
    cov_matrix = (diff.T @ diff) / float(diff.shape[1])
    eig_vals, eig_vects = np.linalg.eig(cov_matrix)
    
    sorted_idx = np.argsort(eig_vals)[::-1]
    top_eig_vects = eig_vects[:, sorted_idx[:K]]
    
    eigenfaces = diff @ top_eig_vects
    
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
    
    trainset_proj = eigenfaces.T @ diff
    
    return eigenfaces, avg_img, trainset_proj, eig_vals[sorted_idx[:K]]

def plot_detailed_reconstruction(original_imgs, eigenfaces, avg_img, K_value=20, img_shape=(50, 50), save_path='detailed_reconstruction.png'):
    """展示多张图像在特定K值下的详细重建效果"""
    num_imgs = min(6, original_imgs.shape[1])  # 最多显示6张图像
    fig, axes = plt.subplots(2, num_imgs, figsize=(3*num_imgs, 6))
    
    if num_imgs == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_imgs):
        # 原始图像
        original = original_imgs[:, i:i+1]
        axes[0, i].imshow(original.reshape(img_shape), cmap='gray')
        axes[0, i].set_title(f'原始图像 {i+1}')
        axes[0, i].axis('off')
        
        # 重建图像
        diff_img = original - avg_img
        projection = eigenfaces[:, :K_value].T @ diff_img
        reconstructed = avg_img + eigenfaces[:, :K_value] @ projection
        
        axes[1, i].imshow(reconstructed.reshape(img_shape), cmap='gray')
        axes[1, i].set_title(f'K={K_value} 重建')
        axes[1, i].axis('off')
    
    plt.suptitle(f'特征脸重建效果对比 (K={K_value})', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"详细重建对比图已保存为: {save_path}")

def plot_reconstruction_quality_analysis(original_img, eigenfaces, avg_img, img_shape=(50, 50), save_path='reconstruction_quality_analysis.png'):
    """分析重建质量随K值变化的详细过程"""
    K_values = [1, 2, 5, 10, 15, 20, 30, 50]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    
    diff_img = original_img - avg_img
    
    for idx, K in enumerate(K_values):
        row = idx // 4
        col = idx % 4
        
        # 投影到K维特征脸空间
        projection = eigenfaces[:, :K].T @ diff_img
        # 重建图像
        reconstructed = avg_img + eigenfaces[:, :K] @ projection
        
        # 计算重建误差
        mse = np.mean((original_img - reconstructed) ** 2)
        
        axes[row, col].imshow(reconstructed.reshape(img_shape), cmap='gray')
        axes[row, col].set_title(f'K={K}\nMSE={mse:.1f}', fontsize=10, pad=8)
        axes[row, col].axis('off')
    
    plt.suptitle('图像重建质量分析：不同K值的重建效果与误差', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"重建质量分析图已保存为: {save_path}")

def plot_eigenface_contribution(original_img, eigenfaces, avg_img, img_shape=(50, 50), save_path='eigenface_contribution.png'):
    """展示各个特征脸对重建的贡献"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 11))
    
    diff_img = original_img - avg_img
    
    # 显示原始图像
    axes[0, 0].imshow(original_img.reshape(img_shape), cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=11, pad=12)
    axes[0, 0].axis('off')
    
    # 显示平均脸
    axes[0, 1].imshow(avg_img.reshape(img_shape), cmap='gray')
    axes[0, 1].set_title('平均脸', fontsize=11, pad=12)
    axes[0, 1].axis('off')
    
    # 显示前8个特征脸的贡献
    cumulative_reconstruction = avg_img.copy()
    
    for i in range(8):
        # 计算第i个特征脸的投影系数
        coeff = eigenfaces[:, i:i+1].T @ diff_img
        # 第i个特征脸的贡献
        contribution = eigenfaces[:, i:i+1] @ coeff
        cumulative_reconstruction += contribution
        
        row = (i + 2) // 5
        col = (i + 2) % 5
        
        axes[row, col].imshow(cumulative_reconstruction.reshape(img_shape), cmap='gray')
        axes[row, col].set_title(f'累积到第{i+1}个特征脸', fontsize=11, pad=12)
        axes[row, col].axis('off')
    
    plt.suptitle('特征脸累积重建过程：逐步添加特征脸的效果', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"特征脸贡献分析图已保存为: {save_path}")

def main():
    """生成额外的重建结果图片"""
    print("=== 生成额外的图像重建结果 ===")
    
    # 检查数据集是否存在
    if not os.path.exists('./yaleBfaces/subset0'):
        print("未找到YaleB人脸数据集，请确保数据集路径正确")
        return
    
    # 加载数据
    print("加载数据...")
    trainset, trainset_labels = get_images('./yaleBfaces/subset0')
    testset, testset_labels = get_images('./yaleBfaces/subset1')
    
    # 训练模型
    print("训练特征脸模型...")
    K = 50  # 使用更多的特征脸
    eigenfaces, avg_img, trainset_proj, eigenvalues = eigen_train(trainset, K)
    
    img_size = int(np.sqrt(trainset.shape[0]))
    
    # 1. 生成详细重建对比图（多张人脸）
    print("生成详细重建对比图...")
    plot_detailed_reconstruction(testset[:, :6], eigenfaces, avg_img, K_value=20, 
                               img_shape=(img_size, img_size), 
                               save_path='detailed_reconstruction.png')
    
    # 2. 生成重建质量分析图
    print("生成重建质量分析图...")
    plot_reconstruction_quality_analysis(testset[:, 0:1], eigenfaces, avg_img, 
                                       img_shape=(img_size, img_size),
                                       save_path='reconstruction_quality_analysis.png')
    
    # 3. 生成特征脸贡献分析图
    print("生成特征脸贡献分析图...")
    plot_eigenface_contribution(testset[:, 0:1], eigenfaces, avg_img, 
                              img_shape=(img_size, img_size),
                              save_path='eigenface_contribution.png')
    
    print("\n额外重建结果图片生成完成！")
    print("新生成的图片文件:")
    print("- detailed_reconstruction.png: 多张人脸的详细重建对比")
    print("- reconstruction_quality_analysis.png: 重建质量随K值变化分析")
    print("- eigenface_contribution.png: 特征脸累积重建过程")

if __name__ == "__main__":
    main()