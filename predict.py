import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import CNNModel
import os

def load_model(model_path="cnn_model.pth"):
    """加载训练好的模型"""
    model = CNNModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"模型加载成功: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    model.eval()
    return model

def preprocess_image(image_path):
    """预处理输入图像"""
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    
    # 定义预处理流程
    # 注意：这里使用与训练时完全相同的标准化参数
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小为28x28
        transforms.ToTensor(),        # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST数据集的标准归一化参数
    ])

    # 应用预处理
    tensor_image = transform(image)

    # 添加批次维度，使其形状为[1, 1, 28, 28]
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def predict_digit(model, image_tensor):
    """使用模型预测数字"""
    with torch.no_grad():
        output = model(image_tensor)
        # 使用log_softmax的逆运算来获得概率
        probabilities = torch.exp(output)
        predicted_digit = output.argmax(dim=1, keepdim=True)
        confidence = probabilities.max().item()

    return predicted_digit.item(), confidence, probabilities

def visualize_prediction(image_path, predicted_digit, confidence, probabilities):
    """可视化预测结果"""
    # 显示原始图像
    image = Image.open(image_path).convert('L')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图像
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'预测数字: {predicted_digit} (置信度: {confidence:.4f})')
    ax1.axis('off')
    
    # 显示概率分布
    digits = list(range(10))
    probs_numpy = probabilities.cpu().numpy()[0]
    bars = ax2.bar(digits, probs_numpy)
    ax2.set_xlabel('数字')
    ax2.set_ylabel('概率')
    ax2.set_title('各类别预测概率')
    ax2.set_xticks(digits)
    
    # 高亮显示预测的数字
    bars[predicted_digit].set_color('red')
    
    # 添加概率值标签
    for i, prob in enumerate(probs_numpy):
        if prob > 0.01:  # 只显示概率大于1%的标签
            ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载模型
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保模型文件 cnn_model.pth 存在")
        return

    # 处理所有数字图片
    found_images = False
    for digit in range(10):
        image_path = f"{digit}.jpg"
        if os.path.exists(image_path):
            found_images = True
            try:
                # 预处理图像
                image_tensor = preprocess_image(image_path)

                # 进行预测
                prediction, confidence, probabilities = predict_digit(model, image_tensor)
                print(f"图像 {image_path}:")
                print(f"  预测数字: {prediction}")
                print(f"  置信度: {confidence:.4f}")
                
                # 显示前3个最高概率
                probs_numpy = probabilities.cpu().numpy()[0]
                top3_indices = np.argsort(probs_numpy)[-3:][::-1]
                print("  前3个预测:")
                for i, idx in enumerate(top3_indices):
                    print(f"    {idx}: {probs_numpy[idx]:.4f}")
                print("-" * 30)

                # 可视化结果（可选，取消注释以显示图表）
                # visualize_prediction(image_path, prediction, confidence, probabilities)

            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
        else:
            print(f"图像文件 {image_path} 不存在")
    
    if not found_images:
        print("没有找到任何数字图片文件 (0.jpg, 1.jpg, ..., 9.jpg)")
        print("请准备一些测试图片或使用 predict_single_image() 函数测试单个图片")
    
def predict_single_image(image_path):
    """预测单个图像"""
    try:
        model = load_model()
        image_tensor = preprocess_image(image_path)
        prediction, confidence, probabilities = predict_digit(model, image_tensor)
        
        print(f"图像: {image_path}")
        print(f"预测数字: {prediction}")
        print(f"置信度: {confidence:.4f}")
        
        # 显示概率分布
        probs_numpy = probabilities.cpu().numpy()[0]
        print("\n所有类别概率:")
        for digit in range(10):
            print(f"  数字 {digit}: {probs_numpy[digit]:.4f}")
        
        # 可视化结果
        visualize_prediction(image_path, prediction, confidence, probabilities)
        
        return prediction, confidence
        
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None

def predict_batch(model, image_paths):
    """批量预测多个图像"""
    predictions = []
    
    for image_path in image_paths:
        try:
            image_tensor = preprocess_image(image_path)
            prediction, confidence, probabilities = predict_digit(model, image_tensor)
            predictions.append((image_path, prediction, confidence))
            print(f"{image_path}: 预测={prediction}, 置信度={confidence:.4f}")
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            predictions.append((image_path, None, None))
    
    return predictions

if __name__ == "__main__":
    main()