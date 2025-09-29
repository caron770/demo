import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from PIL import Image, ImageOps

# 定义与训练时相同的CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 绘图区域类
class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.brush_size = 20
        self.brush_color = Qt.white
        self.last_point = QPoint()

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear_image(self):
        self.image.fill(Qt.black)
        self.update()

    def get_image_data(self):
        # 调整图像大小为28x28像素
        resized_image = self.image.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 转换为灰度图像
        converted_image = resized_image.convertToFormat(QImage.Format_Grayscale8)
        
        # 提取像素数据 - 改进的方法
        width = converted_image.width()
        height = converted_image.height()
        pixels = []
        
        for y in range(height):
            for x in range(width):
                color = converted_image.pixelColor(x, y)
                # 获取亮度值（对于灰度图像，红绿蓝值相同）
                pixels.append(255 - color.red())  # 反转颜色以匹配MNIST格式
        
        # 转换为numpy数组并重塑
        pixel_array = np.array(pixels, dtype=np.float32)
        pixel_array = pixel_array.reshape(28, 28)
        
        # 归一化到[0,1]范围
        pixel_array = pixel_array / 255.0
        
        # 应用与MNIST数据集相同的标准化
        mean = 0.1307
        std = 0.3081
        pixel_array = (pixel_array - mean) / std
        
        # 转换为PyTorch张量并添加通道维度
        tensor = torch.tensor(pixel_array, dtype=torch.float32).unsqueeze(0)
        return tensor

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写数字识别")
        self.setGeometry(100, 100, 400, 400)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 创建绘图区域
        self.drawing_widget = DrawingWidget()
        main_layout.addWidget(self.drawing_widget)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 创建识别按钮
        self.predict_button = QPushButton("识别数字")
        self.predict_button.clicked.connect(self.predict_digit)
        button_layout.addWidget(self.predict_button)
        
        # 创建清除按钮
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(button_layout)
        
        # 创建结果显示标签
        self.result_label = QLabel("请在上方区域绘制一个数字 (0-9)")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: blue;")
        main_layout.addWidget(self.result_label)
        
        # 加载预训练模型
        self.model = CNNModel()
        try:
            self.model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
            self.model.eval()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.result_label.setText("模型加载失败，请确保cnn_model.pth文件存在")

    def predict_digit(self):
        # 获取图像数据
        image_tensor = self.drawing_widget.get_image_data()
        
        # 检查是否有实际绘制内容
        if torch.sum(image_tensor) == 0:
            self.result_label.setText("请先绘制一个数字")
            return
            
        # 使用模型进行预测
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))  # 添加批次维度
            probabilities = torch.exp(output)  # 转换为概率
            prediction = output.argmax(dim=1, keepdim=True)
            confidence = torch.max(probabilities).item()
        
        # 显示结果
        digit = prediction.item()
        self.result_label.setText(f"识别结果: {digit} (置信度: {confidence:.2f})")
        print(f"识别结果: {digit}, 置信度: {confidence:.2f}")

    def clear_canvas(self):
        self.drawing_widget.clear_image()
        self.result_label.setText("请在上方区域绘制一个数字 (0-9)")

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())