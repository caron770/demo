import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model import CNNModel

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
        """获取图像数据并进行预处理，与训练时保持一致"""
        # 将QImage转换为numpy数组
        ptr = self.image.bits()
        ptr.setsize(self.image.byteCount())
        img_array = np.array(ptr).reshape(self.image.height(), self.image.width(), 4)
        
        # 取红色通道作为灰度值（因为我们用白色画笔在黑色背景上绘制）
        gray_array = img_array[:, :, 0]
        
        # 转换为PIL图像
        pil_img = Image.fromarray(gray_array, mode='L')
        
        # 使用与predict.py相同的预处理流程
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 调整大小为28x28
            transforms.ToTensor(),        # 转换为张量，自动归一化到[0,1]
            transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST标准化参数
        ])
        
        # 应用变换
        tensor = transform(pil_img)
        
        print(f"预处理后张量形状: {tensor.shape}")
        print(f"预处理后张量范围: [{torch.min(tensor).item():.4f}, {torch.max(tensor).item():.4f}]")
        print(f"预处理后张量均值: {torch.mean(tensor).item():.4f}, 标准差: {torch.std(tensor).item():.4f}")
        
        return tensor

    def save_current_image(self, filename="debug_image.png"):
        """保存当前绘制的图像用于调试"""
        ptr = self.image.bits()
        ptr.setsize(self.image.byteCount())
        img_array = np.array(ptr).reshape(self.image.height(), self.image.width(), 4)
        gray_array = img_array[:, :, 0]
        pil_img = Image.fromarray(gray_array, mode='L')
        pil_img.save(filename)
        print(f"调试图像已保存至: {filename}")

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写数字识别")
        self.setMinimumSize(760, 540)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1f1f24;
                color: #f5f5f7;
            }
            QLabel {
                color: #f5f5f7;
            }
            QPushButton {
                background-color: #5560ff;
                color: #ffffff;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6b73ff;
            }
            QPushButton:pressed {
                background-color: #3c44d4;
            }
            """
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)
        central_widget.setLayout(main_layout)

        title_label = QLabel("手写数字识别助手")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: 600; letter-spacing: 1px;")
        main_layout.addWidget(title_label)

        subtitle_label = QLabel("在画布中写下 0-9 的任意数字，然后点击“识别数字”查看预测结果")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 13px; color: #b0b0c5;")
        main_layout.addWidget(subtitle_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        main_layout.addLayout(content_layout)

        canvas_card = QFrame()
        canvas_card.setObjectName("canvasCard")
        canvas_card.setStyleSheet(
            """
            QFrame#canvasCard {
                background-color: #272731;
                border-radius: 16px;
            }
            """
        )
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(20, 20, 20, 20)
        canvas_layout.setSpacing(16)
        canvas_card.setLayout(canvas_layout)

        canvas_title = QLabel("绘图区域")
        canvas_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        canvas_layout.addWidget(canvas_title)

        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setStyleSheet(
            "border: 1px dashed #3b3b4a; border-radius: 8px; background-color: #000000;"
        )
        canvas_layout.addWidget(self.drawing_widget, alignment=Qt.AlignCenter)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addStretch(1)

        self.predict_button = QPushButton("识别数字")
        self.predict_button.clicked.connect(self.predict_digit)
        button_layout.addWidget(self.predict_button)

        self.clear_button = QPushButton("清空画布")
        self.clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.clear_button)

        self.save_button = QPushButton("保存调试图像")
        self.save_button.clicked.connect(self.save_debug_image)
        button_layout.addWidget(self.save_button)

        button_layout.addStretch(1)

        canvas_layout.addLayout(button_layout)
        content_layout.addWidget(canvas_card, 1)

        info_card = QFrame()
        info_card.setObjectName("infoCard")
        info_card.setStyleSheet(
            """
            QFrame#infoCard {
                background-color: #272731;
                border-radius: 16px;
            }
            """
        )
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(20, 20, 20, 20)
        info_layout.setSpacing(16)
        info_card.setLayout(info_layout)

        status_title = QLabel("识别状态")
        status_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        info_layout.addWidget(status_title)

        self.status_label = QLabel("模型加载中……")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; color: #b0b0c5;")
        info_layout.addWidget(self.status_label)

        result_title = QLabel("预测结果")
        result_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        info_layout.addWidget(result_title)

        self.result_label = QLabel("—")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 48px; font-weight: 700; color: #7f89ff;")
        info_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("准备绘制数字")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("font-size: 13px; color: #b0b0c5;")
        info_layout.addWidget(self.confidence_label)

        prob_title = QLabel("Top 3 置信度")
        prob_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        info_layout.addWidget(prob_title)

        self.prob_label = QLabel("等待输入……")
        self.prob_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.prob_label.setTextFormat(Qt.RichText)
        self.prob_label.setWordWrap(True)
        self.prob_label.setStyleSheet("font-size: 13px; color: #d9d9e1;")
        info_layout.addWidget(self.prob_label)

        info_layout.addStretch(1)
        content_layout.addWidget(info_card, 1)

        # 加载预训练模型
        self.model = None
        self.load_model()

    def load_model(self):
        """加载预训练模型"""
        try:
            self.model = CNNModel()
            self.model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
            self.model.eval()
            print("模型加载成功")
            self.status_label.setText("模型加载成功，开始在左侧画布绘制数字吧！")
            self.result_label.setText("—")
            self.confidence_label.setText("在画布中写下 0-9 的任意数字，然后点击“识别数字”。")
            self.prob_label.setText("等待输入……")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.status_label.setText("模型加载失败，请确保 cnn_model.pth 文件存在")
            self.result_label.setText("!")
            self.confidence_label.setText("模型未就绪，暂时无法识别。")
            self.model = None

    def predict_digit(self):
        """预测绘制的数字"""
        if self.model is None:
            self.status_label.setText("模型未加载，无法进行预测")
            self.confidence_label.setText("请先加载模型后再尝试。")
            return
            
        # 获取图像数据
        image_tensor = self.drawing_widget.get_image_data()
        
        # 检查是否有实际绘制内容
        # 由于标准化后的值可能为负，我们检查是否有显著变化
        if torch.std(image_tensor).item() < 0.1:
            self.status_label.setText("画布中尚未检测到有效笔迹")
            self.result_label.setText("—")
            self.confidence_label.setText("请在左侧画布绘制一个清晰的数字。")
            self.prob_label.setText("等待输入……")
            return
            
        # 使用模型进行预测
        self.status_label.setText("正在识别……")
        with torch.no_grad():
            # 添加批次维度，使其形状为[1, 1, 28, 28]
            input_tensor = image_tensor.unsqueeze(0)
            print(f"模型输入张量形状: {input_tensor.shape}")
            
            # 获取模型输出
            output = self.model(input_tensor)
            print(f"模型原始输出: {output}")
            
            # 转换为概率分布
            probabilities = torch.exp(output)  # 由于模型输出是log_softmax
            print(f"概率分布: {probabilities}")
            
            # 获取预测结果
            prediction = output.argmax(dim=1)
            confidence = torch.max(probabilities).item()
        
        # 显示结果
        digit = prediction.item()
        self.result_label.setText(str(digit))
        self.status_label.setText("识别完成")
        self.confidence_label.setText(f"最高置信度：{confidence:.1%}")
        
        # 显示概率分布
        probs = probabilities.cpu().numpy()[0]
        top3_indices = np.argsort(probs)[-3:][::-1]
        rows = []
        for idx in top3_indices:
            rows.append(
                f"<tr>"
                f"<td style='padding:4px 24px 4px 0; color:#b0b0c5;'>数字 {idx}</td>"
                f"<td style='padding:4px 0; color:#f5f5f7;'>{probs[idx]:.1%}</td>"
                f"</tr>"
            )
        table_html = (
            "<table style='border-collapse:collapse; font-size:13px;'>"
            + "".join(rows)
            + "</table>"
        )
        self.prob_label.setText(table_html)
        
        print(f"识别结果: {digit}, 最高概率: {confidence:.4f}")
        print("完整概率分布:")
        for i, prob in enumerate(probs):
            print(f"  数字 {i}: {prob:.4f}")
        print("-" * 50)

    def save_debug_image(self):
        """保存当前绘制的图像用于调试"""
        self.drawing_widget.save_current_image()
        self.status_label.setText("调试图像已保存（debug_image.png）")

    def clear_canvas(self):
        """清除绘制内容"""
        self.drawing_widget.clear_image()
        self.result_label.setText("—")
        self.status_label.setText("画布已清空，重新绘制一个数字吧！")
        self.confidence_label.setText("准备绘制数字")
        self.prob_label.setText("等待输入……")

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
