import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import os


class VesselUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置图标
        icon = QIcon('icon.ico')
        self.setWindowIcon(icon)

        self.setWindowTitle('血管分割系统')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # 创建选项卡
        tab_widget = QTabWidget()

        # 传统方法标签页
        traditional_tab = QWidget()
        traditional_layout = QVBoxLayout(traditional_tab)

        # U-Net方法标签页
        unet_tab = QWidget()
        unet_layout = QVBoxLayout(unet_tab)

        # 添加logo
        logo_label = QLabel()
        logo_pixmap = QPixmap('logo.jpg')  # 替换为实际的logo图片路径
        scaled_pixmap = logo_pixmap.scaled(250, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        # 传统方法的控件
        self.load_btn = QPushButton('加载图像')
        self.load_mask_btn = QPushButton('加载掩膜')
        self.process_btn = QPushButton('传统方法处理')
        self.save_btn = QPushButton('保存结果')

        self.step_combo = QComboBox()
        self.step_combo.addItems([
            '原始图像',
            '掩膜图像',
            '高斯平滑',
            '限制对比度自适应直方图均衡化',
            '同态滤波',
            '匹配滤波',
            '灰度拉伸',
            'Otsu阈值分割',

        ])

        # Parameters
        param_group = QGroupBox('参数设置')
        param_layout = QFormLayout()

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setValue(1.5)
        self.gamma_spin.setSingleStep(0.1)

        self.m_spin = QDoubleSpinBox()
        self.m_spin.setRange(0.01, 1.0)
        self.m_spin.setValue(30.0 / 255)
        self.m_spin.setSingleStep(0.01)

        self.e_spin = QDoubleSpinBox()
        self.e_spin.setRange(1.0, 20.0)
        self.e_spin.setValue(8.0)
        self.e_spin.setSingleStep(0.5)

        param_layout.addRow('伽马值:', self.gamma_spin)
        param_layout.addRow('M值:', self.m_spin)
        param_layout.addRow('E值:', self.e_spin)
        param_group.setLayout(param_layout)

        # U-Net方法的控件
        self.load_unet_image_btn = QPushButton('加载待测图像')
        self.load_unet_fov_btn = QPushButton('加载FOV掩膜')
        self.process_unet_btn = QPushButton('U-Net处理')
        self.save_unet_btn = QPushButton('保存U-Net结果')

        # 添加控件到传统方法布局
        traditional_layout.addWidget(logo_label)  # 添加logo
        traditional_layout.addWidget(self.load_btn)
        traditional_layout.addWidget(self.load_mask_btn)
        traditional_layout.addWidget(self.step_combo)
        traditional_layout.addWidget(param_group)
        traditional_layout.addWidget(self.process_btn)
        traditional_layout.addWidget(self.save_btn)
        traditional_layout.addStretch()

        # 添加控件到U-Net布局
        unet_layout.addWidget(self.load_unet_image_btn)
        unet_layout.addWidget(self.load_unet_fov_btn)
        unet_layout.addWidget(self.process_unet_btn)
        unet_layout.addWidget(self.save_unet_btn)
        unet_layout.addStretch()

        # 添加标签页
        tab_widget.addTab(traditional_tab, "传统方法")
        tab_widget.addTab(unet_tab, "U-Net方法")

        # 添加标签页到左侧面板
        left_layout.addWidget(tab_widget)

        # Right panel for image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Add scroll area for image
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)

        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(scroll)

        # Status bar
        self.statusBar().showMessage('就绪')

    def display_image(self, img):
        if img is None:
            return

        # Convert numpy array to correct format
        if isinstance(img, np.ndarray):
            height, width = img.shape[:2]

            if len(img.shape) == 2:  # Grayscale image
                # Convert to copy of data
                img = img.copy()
                qimg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
            else:  # Color image
                # Convert BGR to RGB and get copy of data
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
                bytes_per_line = 3 * width
                qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)

            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)

            self.image_label.setPixmap(scaled_pixmap)

    def show_error_message(self, message):
        QMessageBox.critical(self, '错误', message)