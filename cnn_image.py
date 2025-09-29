import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一个卷积层: 输入通道1(灰度图), 输出通道32, 卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # 第二个卷积层: 输入通道32, 输出通道64, 卷积核大小5x5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # Dropout层防止过拟合
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全连接层
        self.fc1 = nn.Linear(1024, 128)  # 1024 = 64 * 4 * 4 (经过两次卷积和池化后)
        self.fc2 = nn.Linear(128, 10)    # 10个类别(0-9)

    def forward(self, x):
        # 第一个卷积块: 卷积 -> ReLU -> 最大池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块: 卷积 -> ReLU -> 最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Dropout防止过拟合
        x = self.dropout1(x)
        
        # 展平操作
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # 输出层使用log_softmax
        output = F.log_softmax(x, dim=1)
        return output

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载数据集
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 预测单张图片的函数
def predict_image(model, device, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加批次维度
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
        return prediction.item()

# 显示一些测试图像及其预测结果
def show_predictions(model, device, test_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True)
            
            # 显示前num_images个图像
            fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            for i in range(num_images):
                row = i // 5
                col = i % 5
                # 将标准化的图像恢复到[0,1]范围
                img = data[i].cpu().numpy().squeeze()
                img = (img * 0.3081) + 0.1307
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f'Predicted: {predictions[i].item()}, Actual: {target[i].item()}')
                axes[row, col].axis('off')
            plt.tight_layout()
            plt.show()
            break  # 只显示一个批次

# 主程序
if __name__ == '__main__':
    # 训练模型
    epochs = 3
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), "cnn_model.pth")
    print("模型已保存为 cnn_model.pth")
    
    # 显示一些预测结果
    print("\n显示一些预测结果:")
    show_predictions(model, device, test_loader)