#!/usr/bin/env python3
"""
调试脚本 - 用于诊断手写数字识别的问题
"""

import torch
import numpy as np
from torchvision import transforms, datasets
from model import CNNModel
import matplotlib.pyplot as plt
from PIL import Image
import os

def test_model_on_mnist():
    """在MNIST测试集上测试模型性能"""
    print("=== 测试模型在MNIST数据集上的性能 ===")
    
    # 加载模型
    model = CNNModel()
    try:
        model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
        model.eval()
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 加载MNIST测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
        print("✓ MNIST测试数据加载成功")
    except Exception as e:
        print(f"✗ MNIST数据加载失败: {e}")
        return
    
    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if total >= 1000:  # 只测试前1000个样本
                break
    
    accuracy = 100 * correct / total
    print(f"✓ 模型在MNIST测试集上的准确率: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy > 90  # 如果准确率低于90%，说明模型有问题

def visualize_mnist_samples():
    """可视化MNIST样本及其预处理过程"""
    print("\n=== 可视化MNIST样本 ===")
    
    # 加载模型
    model = CNNModel()
    try:
        model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # 选择几个样本进行可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(10):
        row = i // 5
        col = i % 5
        
        # 获取样本
        sample, label = test_dataset[i]
        
        # 预测
        with torch.no_grad():
            output = model(sample.unsqueeze(0))
            prediction = output.argmax(dim=1).item()
        
        # 显示图像
        # 反标准化以便显示
        img = sample.squeeze() * 0.3081 + 0.1307
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'真实: {label}, 预测: {prediction}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples_test.png')
    plt.show()
    print("✓ MNIST样本可视化完成，保存为 mnist_samples_test.png")

def test_preprocessing_consistency():
    """测试预处理的一致性"""
    print("\n=== 测试预处理一致性 ===")
    
    # 创建一个简单的测试图像（白色数字7在黑色背景上）
    test_img = np.zeros((280, 280), dtype=np.uint8)
    # 画一个简单的7
    test_img[50:80, 50:200] = 255  # 水平线
    test_img[80:200, 170:200] = 255  # 竖直线
    
    # 保存测试图像
    test_pil = Image.fromarray(test_img, mode='L')
    test_pil.save('test_7.png')
    print("✓ 创建测试图像 test_7.png")
    
    # 方法1：使用predict.py的预处理方式
    from predict import preprocess_image
    tensor1 = preprocess_image('test_7.png')
    
    # 方法2：模拟UI的预处理方式
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor2 = transform(test_pil).unsqueeze(0)
    
    # 比较两种方法
    print(f"predict.py方法张量形状: {tensor1.shape}")
    print(f"UI方法张量形状: {tensor2.shape}")
    print(f"两种方法是否一致: {torch.allclose(tensor1, tensor2, atol=1e-6)}")
    
    if not torch.allclose(tensor1, tensor2, atol=1e-6):
        print("⚠️ 预处理方法不一致！")
        print(f"差异: {torch.max(torch.abs(tensor1 - tensor2)).item()}")
    else:
        print("✓ 预处理方法一致")
    
    # 测试预测
    model = CNNModel()
    try:
        model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            output1 = model(tensor1)
            output2 = model(tensor2)
            pred1 = output1.argmax(dim=1).item()
            pred2 = output2.argmax(dim=1).item()
        
        print(f"predict.py方法预测: {pred1}")
        print(f"UI方法预测: {pred2}")
        print(f"预测是否一致: {pred1 == pred2}")
        
    except Exception as e:
        print(f"预测测试失败: {e}")

def check_model_structure():
    """检查模型结构"""
    print("\n=== 检查模型结构 ===")
    
    model = CNNModel()
    print("模型结构:")
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        print(f"✓ 模型前向传播正常，输出形状: {output.shape}")
        print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 检查是否是log概率
        probs = torch.exp(output)
        print(f"转换为概率后的和: {probs.sum().item():.4f}")
        
        if abs(probs.sum().item() - 1.0) < 1e-5:
            print("✓ 输出是有效的log概率分布")
        else:
            print("⚠️ 输出可能不是有效的log概率分布")
            
    except Exception as e:
        print(f"✗ 模型前向传播失败: {e}")

def main():
    """主函数"""
    print("开始模型诊断...\n")
    
    # 检查必要文件
    if not os.path.exists("cnn_model.pth"):
        print("✗ 模型文件 cnn_model.pth 不存在")
        return
    
    if not os.path.exists("data/MNIST"):
        print("✗ MNIST数据集不存在")
        return
    
    # 运行所有测试
    check_model_structure()
    model_ok = test_model_on_mnist()
    test_preprocessing_consistency()
    
    if model_ok:
        visualize_mnist_samples()
    
    print("\n=== 诊断完成 ===")
    print("如果发现问题，请检查:")
    print("1. 模型训练是否正确完成")
    print("2. 预处理步骤是否与训练时一致")
    print("3. 输入图像的格式和颜色是否正确")

if __name__ == "__main__":
    main()