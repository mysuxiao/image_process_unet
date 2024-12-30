import sys
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
from ui import VesselUI
from handle import VesselHandler
import torch
import torch.backends.cudnn as cudnn
from models import LadderNet
from unet import test_single_image


class VesselSegmentation:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ui = VesselUI()
        self.handler = VesselHandler()
        self.setup_connections()
        self.setup_unet()
        self.ui.show()

        # U-Net相关变量
        self.unet_image_path = None
        self.unet_fov_path = None

    def setup_unet(self):
        try:
            # 初始化U-Net模型
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(self.device)
            cudnn.benchmark = True

            # 加载模型权重
            checkpoint = torch.load('./best_model_1.pth')  # 替换为实际的模型路径
            self.net.load_state_dict(checkpoint['net'])
        except Exception as e:
            self.ui.show_error_message(f"U-Net模型加载失败: {str(e)}")

    def setup_connections(self):
        # 传统方法的连接
        self.ui.load_btn.clicked.connect(self.load_image)
        self.ui.load_mask_btn.clicked.connect(self.load_mask)
        self.ui.process_btn.clicked.connect(self.process_image)
        self.ui.save_btn.clicked.connect(self.save_result)
        self.ui.step_combo.currentTextChanged.connect(self.update_display)

        # U-Net方法的连接
        self.ui.load_unet_image_btn.clicked.connect(self.load_unet_image)
        self.ui.load_unet_fov_btn.clicked.connect(self.load_unet_fov)
        self.ui.process_unet_btn.clicked.connect(self.process_unet)
        self.ui.save_unet_btn.clicked.connect(self.save_unet_result)

        # Connect handler signals
        self.handler.error_occurred.connect(self.ui.show_error_message)

    def load_mask(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select Mask",
            "",
            "Image Files (*.gif *.png *.jpg *.bmp);;All Files (*.*)"
        )
        if filename:
            self.handler.load_mask(filename)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif);;All Files (*.*)"
        )
        if filename:
            if self.handler.load_image(filename):
                self.update_display()
                self.ui.statusBar().showMessage('Image loaded successfully')
            else:
                self.ui.statusBar().showMessage('Failed to load image')

    def load_unet_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select Image for U-Net",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif);;All Files (*.*)"
        )
        if filename:
            self.unet_image_path = filename
            self.ui.statusBar().showMessage('U-Net图像已加载')
            # 显示加载的图像
            img = cv2.imread(filename)
            self.ui.display_image(img)

    def load_unet_fov(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Select FOV Mask for U-Net",
            "",
            "Image Files (*.gif *.png *.jpg *.bmp);;All Files (*.*)"
        )
        if filename:
            self.unet_fov_path = filename
            self.ui.statusBar().showMessage('FOV掩膜已加载')

    def process_image(self):
        gamma = self.ui.gamma_spin.value()
        m = self.ui.m_spin.value()
        e = self.ui.e_spin.value()

        if self.handler.process_image(gamma, m, e):
            self.update_display()
            self.ui.statusBar().showMessage('Processing completed')
        else:
            self.ui.statusBar().showMessage('Processing failed')

    def process_unet(self):
        if not self.unet_image_path:
            self.ui.show_error_message("请先加载待测图像")
            return

        try:
            save_path = "temp_result.png"  # 临时保存路径
            result = test_single_image(self.unet_image_path, self.unet_fov_path,
                                       self.net, save_path)
            if result is not None:
                self.ui.display_image(result)
                self.ui.statusBar().showMessage('U-Net处理完成')
                self.unet_result = result
            else:
                self.ui.statusBar().showMessage('U-Net处理失败')
        except Exception as e:
            self.ui.show_error_message(f"U-Net处理错误: {str(e)}")

    def update_display(self):
        step = self.ui.step_combo.currentText()
        result = self.handler.get_result(step)
        if result is not None:
            self.ui.display_image(result)

    def save_result(self):
        filename, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Save Result",
            "",
            "PNG Files (*.png);;All Files (*.*)"
        )
        if filename:
            if self.handler.save_result(filename):
                self.ui.statusBar().showMessage('Result saved successfully')
            else:
                self.ui.statusBar().showMessage('Failed to save result')

    def save_unet_result(self):
        if not hasattr(self, 'unet_result'):
            self.ui.show_error_message("没有可保存的U-Net结果")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Save U-Net Result",
            "",
            "PNG Files (*.png);;All Files (*.*)"
        )
        if filename:
            try:
                cv2.imwrite(filename, self.unet_result)
                self.ui.statusBar().showMessage('U-Net结果保存成功')
            except Exception as e:
                self.ui.show_error_message(f"保存失败: {str(e)}")

    def run(self):
        return self.app.exec_()


if __name__ == '__main__':
    app = VesselSegmentation()
    sys.exit(app.run())