import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import json
import base64
from io import BytesIO
from PIL import Image

def decode_image_from_base64(base64_string):
    """从base64字符串解码图像"""
    # 移除data:image/png;base64,前缀
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # 解码base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    # 转换为numpy数组
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # 移除alpha通道
    
    return img_array.astype(np.float64) / 255.0

def encode_image_to_base64(img_array):
    """将图像数组编码为base64字符串"""
    # 转换为0-255范围
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # 转换为PIL图像
    image = Image.fromarray(img_uint8)
    
    # 编码为base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def add_noise_to_image(img, noise_ratio):
    """向图像添加噪声"""
    noise_img = np.copy(img)
    h, w, c = img.shape
    
    # 生成随机掩码
    mask = np.random.rand(h, w) < noise_ratio
    
    # 将噪声点设为0（黑色）
    for channel in range(c):
        noise_img[mask, channel] = 0
    
    return noise_img, mask

def restore_image_regression(noise_img, mask, region_size=2):
    """使用线性回归恢复图像"""
    restored_img = np.copy(noise_img)
    h, w, c = noise_img.shape
    
    # 对每个颜色通道进行恢复
    for channel in range(c):
        channel_data = noise_img[:, :, channel]
        
        # 找到所有噪声点
        noise_points = np.where(mask)
        
        for i in range(len(noise_points[0])):
            y, x = noise_points[0][i], noise_points[1][i]
            
            # 收集邻域非噪声点
            X_train = []
            y_train = []
            
            for dy in range(-region_size, region_size + 1):
                for dx in range(-region_size, region_size + 1):
                    ny, nx = y + dy, x + dx
                    
                    # 检查边界
                    if 0 <= ny < h and 0 <= nx < w:
                        # 如果不是噪声点
                        if not mask[ny, nx]:
                            X_train.append([nx, ny])  # 坐标作为特征
                            y_train.append(channel_data[ny, nx])  # 像素值作为目标
            
            # 如果有足够的训练数据
            if len(X_train) >= 3:  # 至少需要3个点进行线性回归
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # 训练线性回归模型
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # 预测噪声点的值
                predicted_value = model.predict([[x, y]])[0]
                
                # 确保值在[0,1]范围内
                predicted_value = np.clip(predicted_value, 0, 1)
                
                restored_img[y, x, channel] = predicted_value
            else:
                # 如果邻域点不足，使用简单平均
                if len(y_train) > 0:
                    restored_img[y, x, channel] = np.mean(y_train)
    
    return restored_img

def calculate_rmse(original_img, restored_img):
    """计算RMSE误差"""
    mse = np.mean((original_img - restored_img) ** 2)
    return np.sqrt(mse)

def main():
    try:
        # 从命令行参数读取输入
        if len(sys.argv) != 4:
            raise ValueError("参数数量不正确")
        
        original_base64 = sys.argv[1]
        noise_ratio = float(sys.argv[2])
        region_size = int(sys.argv[3])
        
        # 解码原始图像
        original_img = decode_image_from_base64(original_base64)
        
        # 添加噪声
        noise_img, mask = add_noise_to_image(original_img, noise_ratio)
        
        # 恢复图像
        restored_img = restore_image_regression(noise_img, mask, region_size)
        
        # 计算误差
        rmse = calculate_rmse(original_img, restored_img)
        
        # 编码结果图像
        noise_base64 = encode_image_to_base64(noise_img)
        restored_base64 = encode_image_to_base64(restored_img)
        
        # 返回结果
        result = {
            "success": True,
            "noise_image": noise_base64,
            "restored_image": restored_base64,
            "rmse": float(rmse),
            "noise_ratio": noise_ratio,
            "region_size": region_size
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()