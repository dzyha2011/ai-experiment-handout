from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import numpy as np
import image_restoration
import json

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 图像恢复处理器已通过导入image_restoration模块获得

@app.route('/')
def index():
    """提供主页HTML文件"""
    return send_file('第三章 回归分析实验.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件服务"""
    return send_from_directory('.', filename)

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """处理图像上传"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': '没有提供图像数据'}), 400
        
        # 解码base64图像
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 转换为PIL图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调整图像大小（如果太大）
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 转换回base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        processed_image = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{processed_image}',
            'width': image.size[0],
            'height': image.size[1]
        })
    
    except Exception as e:
        return jsonify({'error': f'图像处理失败: {str(e)}'}), 500

@app.route('/api/add_noise', methods=['POST'])
def add_noise():
    """为图像添加噪声"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': '没有提供图像数据'}), 400
        
        image_data = data['image']
        noise_ratio = float(data.get('noise_ratio', 0.3))
        
        # 解码图像
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 添加噪声
        noisy_image_data = image_processor.add_noise(image_data, noise_ratio)
        noisy_image = Image.open(io.BytesIO(base64.b64decode(noisy_image_data)))
        
        # 转换回base64
        buffer = io.BytesIO()
        noisy_image.save(buffer, format='PNG')
        result_image = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'noisy_image': f'data:image/png;base64,{result_image}',
            'noise_ratio': noise_ratio
        })
    
    except Exception as e:
        return jsonify({'error': f'添加噪声失败: {str(e)}'}), 500

@app.route('/api/restore_image', methods=['POST'])
def restore_image():
    try:
        data = request.get_json()
        image_data = data.get('image')
        noise_ratio = data.get('noise_ratio', 0.3)
        region_size = data.get('region_size', 2)
        
        if not image_data:
            return jsonify({'success': False, 'error': '未提供图像数据'}), 400
        
        # 解码原始图像
        original_img = image_restoration.decode_image_from_base64(image_data)
        
        # 添加噪声
        noise_img, mask = image_restoration.add_noise_to_image(original_img, noise_ratio)
        
        # 恢复图像
        restored_img = image_restoration.restore_image_regression(noise_img, mask, region_size)
        
        # 计算误差
        rmse = image_restoration.calculate_rmse(original_img, restored_img)
        
        # 编码结果图像
        noise_base64 = image_restoration.encode_image_to_base64(noise_img)
        restored_base64 = image_restoration.encode_image_to_base64(restored_img)
        
        result = {
            'success': True,
            'noise_image': noise_base64,
            'restored_image': restored_base64,
            'rmse': float(rmse),
            'noise_ratio': noise_ratio,
            'region_size': region_size
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'图像恢复失败: {str(e)}'
        }), 500

@app.route('/api/calculate_error', methods=['POST'])
def calculate_error():
    """计算恢复误差"""
    try:
        data = request.get_json()
        if 'original_image' not in data or 'restored_image' not in data:
            return jsonify({'error': '缺少必要的图像数据'}), 400
        
        original_data = data['original_image']
        restored_data = data['restored_image']
        
        # 解码图像数据
        if original_data.startswith('data:image'):
            original_data = original_data.split(',')[1]
        if restored_data.startswith('data:image'):
            restored_data = restored_data.split(',')[1]
        
        # 计算RMSE
        rmse = image_processor.calculate_rmse(original_data, restored_data)
        
        return jsonify({
            'success': True,
            'rmse': rmse
        })
    
    except Exception as e:
        return jsonify({'error': f'误差计算失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("启动Flask服务器...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)