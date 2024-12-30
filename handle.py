import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image
from vessel_seg import *
class VesselHandler(QObject):
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.image = None
        self.mask = None
        self.groundtruth = None
        self.results = {}

    def load_image(self, filename):
        try:
            self.image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if self.image is None:
                raise Exception("Failed to load image")
            self.results['原始图像'] = self.image
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def load_mask(self, filename):
        try:
            mask_pil = Image.open(filename)
            self.mask = np.array(mask_pil)
            if len(self.mask.shape) > 2:
                self.mask = cv2.cvtColor(self.mask, cv2.COLOR_RGB2GRAY)
            self.results['掩膜图像'] = self.mask

            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def process_image(self, gamma=1.5, m=30.0/255, e=8.0):
        try:
            if self.image is None:
                raise Exception("No image loaded")
            if self.mask is None:
                raise Exception("No mask loaded")

            # Extract green channel
            grayImg = cv2.split(self.image)[1]

            # Apply mask
            maskedImg = cv2.bitwise_and(grayImg, grayImg, mask=self.mask)

            # Gaussian blur
            blurImg = cv2.GaussianBlur(maskedImg, (5, 5), 0)
            self.results['高斯平滑'] = blurImg

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
            claheImg = clahe.apply(blurImg)
            self.results['限制对比度自适应直方图均衡化'] = claheImg

            # Homomorphic filtering
            homoImg = homofilter(blurImg)
            self.results['同态滤波'] = homoImg

            # Matched filtering
            preMFImg = adjust_gamma(claheImg, gamma=gamma)
            filters = build_filters2()
            gaussMFImg = process(preMFImg, filters)
            gaussMFImg_mask = pass_mask(self.mask, gaussMFImg)
            self.results['匹配滤波'] = gaussMFImg_mask

            # Gray stretch
            grayStretchImg = grayStretch(gaussMFImg_mask, m=m, e=e)
            self.results['灰度拉伸'] = grayStretchImg

            # Final thresholding
            ret1, predictImg = cv2.threshold(grayStretchImg, 30, 255, cv2.THRESH_OTSU)
            predictImg = cv2.bitwise_and(predictImg, predictImg, mask=self.mask)
            self.results['Otsu阈值分割'] = predictImg

            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def get_result(self, step):
        return self.results.get(step)

    def save_result(self, filename):
        try:
            if '最终结果' not in self.results:
                raise Exception("No result to save")
            cv2.imwrite(filename, self.results['最终结果'])
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False