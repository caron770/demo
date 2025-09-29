#!/usr/bin/env python3
"""
创建类似MNIST风格的测试图像
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

def create_mnist_style_digit(digit, size=(28, 28), noise_level=0.1):
    """创建类似MNIST风格的数字图像"""
    
    # 创建黑色背景 (MNIST风格)
    img = Image.new('L', (100, 100), color=0)
    draw = ImageDraw.Draw(img)
    
    # 尝试使用不同的字体
    font_size = 60
    try:
        # 尝试使用手写风格的字体
        font = ImageFont.truetype("/System/Library/Fonts/Marker Felt.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Comic Sans MS.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    
    # 绘制白色数字
    text = str(digit)
    
    # 计算文本位置使其居中
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (100 - text_width) // 2
    y = (100 - text_height) // 2
    
    # 绘制数字 (白色 = 255)
    draw.text((x, y), text, fill=255, font=font)
    
    # 调整大小到28x28
    img = img.resize(size, Image.LANCZOS)
    
    # 转换为numpy数组并添加一些噪声
    img_array = np.array(img, dtype=np.float32)
    
    # 添加轻微噪声使其更像手写
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        img_array = img_array + noise
        img_array = np.clip(img_array, 0, 255)
    
    # 应用轻微模糊使边缘更平滑
    img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img

def create_handwritten_style_digit(digit, size=(28, 28)):
    """使用简单几何形状创建更像手写的数字"""
    img = Image.new('L', (28, 28), color=0)
    draw = ImageDraw.Draw(img)
    
    # 定义各个数字的绘制方式
    if digit == 0:
        draw.ellipse([4, 4, 23, 23], outline=255, width=2)
    elif digit == 1:
        draw.line([14, 4, 14, 23], fill=255, width=2)
        draw.line([12, 6, 14, 4], fill=255, width=1)
    elif digit == 2:
        draw.arc([4, 4, 23, 15], 0, 180, fill=255, width=2)
        draw.line([23, 12, 4, 23], fill=255, width=2)
        draw.line([4, 23, 23, 23], fill=255, width=2)
    elif digit == 3:
        draw.arc([4, 4, 23, 15], 0, 180, fill=255, width=2)
        draw.arc([4, 13, 23, 24], 0, 180, fill=255, width=2)
        draw.line([12, 13, 20, 13], fill=255, width=1)
    elif digit == 4:
        draw.line([6, 4, 6, 15], fill=255, width=2)
        draw.line([18, 4, 18, 23], fill=255, width=2)
        draw.line([6, 15, 23, 15], fill=255, width=2)
    elif digit == 5:
        draw.line([4, 4, 20, 4], fill=255, width=2)
        draw.line([4, 4, 4, 13], fill=255, width=2)
        draw.arc([4, 13, 23, 24], 0, 180, fill=255, width=2)
    elif digit == 6:
        draw.arc([4, 4, 23, 24], 90, 270, fill=255, width=2)
        draw.ellipse([4, 13, 23, 24], outline=255, width=2)
    elif digit == 7:
        draw.line([4, 4, 23, 4], fill=255, width=2)
        draw.line([20, 4, 8, 23], fill=255, width=2)
    elif digit == 8:
        draw.ellipse([4, 4, 23, 15], outline=255, width=2)
        draw.ellipse([4, 13, 23, 24], outline=255, width=2)
    elif digit == 9:
        draw.ellipse([4, 4, 23, 15], outline=255, width=2)
        draw.arc([4, 4, 23, 24], 270, 90, fill=255, width=2)
    
    return img

def main():
    """创建MNIST风格的测试图像"""
    print("正在创建MNIST风格的测试图像...")
    
    # 创建两套图像：字体版本和手绘版本
    for digit in range(10):
        # 字体版本
        img1 = create_mnist_style_digit(digit)
        filename1 = f"mnist_style_{digit}.jpg"
        img1.save(filename1)
        print(f"✓ 创建字体版本: {filename1}")
        
        # 手绘版本
        img2 = create_handwritten_style_digit(digit)
        filename2 = f"handwritten_{digit}.jpg"
        img2.save(filename2)
        print(f"✓ 创建手绘版本: {filename2}")
    
    print("\n测试图像创建完成！")
    print("请使用以下命令测试这些图像:")
    print("python predict.py")
    print("或者使用predict_single_image函数测试单个图像")

if __name__ == "__main__":
    main()