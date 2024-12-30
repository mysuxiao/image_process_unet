import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import os
from PIL import Image


def my_PreProc(data):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    # black-white conversion
    train_imgs = rgb2gray(data)
    # my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  # reduce to 0-1 range
    return train_imgs


def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def kill_border(data, FOVs):
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if not pixel_inside_FOV(i, x, y, FOVs):
                    data[i, :, y, x] = 0.0


def pixel_inside_FOV(i, x, y, FOVs):
    assert (len(FOVs.shape) == 4)  # 4D arrays
    assert (FOVs.shape[1] == 1)
    if (x >= FOVs.shape[3] or y >= FOVs.shape[2]):  # Pixel position is out of range
        return False
    return FOVs[i, 0, y, x] > 0  # 0==black pixels


def read_gif_image(gif_path):
    """使用PIL读取GIF图像并转换为numpy数组"""
    try:
        # 使用PIL打开GIF图像
        img = Image.open(gif_path)
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        # 转换为numpy数组
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error reading GIF image: {str(e)}")
        return None


def preprocess_single_image(img_path, fov_path=None, target_size=(576, 576)):
    # 读取并检查原始图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取FOV，如果是GIF格式使用PIL读取
    if fov_path and os.path.exists(fov_path):
        if fov_path.lower().endswith('.gif'):
            fov = read_gif_image(fov_path)
        else:
            fov = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)

        if fov is None:
            print(f"Warning: Failed to load FOV from {fov_path}, creating default FOV")
            fov = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    else:
        fov = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    # 调整图像尺寸为模型要求的尺寸
    img = cv2.resize(img, target_size)
    fov = cv2.resize(fov, target_size)

    # 调整维度以匹配批处理格式 (1, 3, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    fov = np.expand_dims(fov, 0)
    fov = np.expand_dims(fov, 0)

    # 使用相同的预处理流程
    img = my_PreProc(img)

    return img, fov

def test_single_image(img_path, fov_path, net, save_path, original_img=None):
    # 获取原始图像尺寸（如果提供）
    if original_img is None:
        original_img = cv2.imread(img_path)
    original_size = (original_img.shape[1], original_img.shape[0])  # (width, height)

    # 预处理
    img, fov = preprocess_single_image(img_path, fov_path)

    # 转换为tensor
    img_tensor = torch.FloatTensor(img).cuda()

    # 网络预测
    net.eval()
    with torch.no_grad():
        output = net(img_tensor)
        # 获取第二个通道的输出
        pred = output[:, 1].cpu().numpy()
        pred = np.expand_dims(pred, axis=1)

    # 处理预测结果
    kill_border(pred, fov)

    # 二值化处理
    pred = (pred > 0.5).astype(np.uint8) * 255

    # 移除批次和通道维度
    pred = pred[0, 0]

    # 将预测结果调整回原始图像大小
    pred = cv2.resize(pred, original_size)

    # 保存结果
    cv2.imwrite(save_path, pred)

    return pred


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    from models import LadderNet  # 确保models.py在正确的路径下

    net = LadderNet(inplanes=1, num_classes=2, layers=3, filters=16).to(device)
    cudnn.benchmark = True

    # 加载模型权重
    checkpoint = torch.load('./best_model_1.pth')  # 替换为实际的模型路径
    net.load_state_dict(checkpoint['net'])

    # 测试图像路径
    img_path = '21_training.tif'  # 替换为实际的图像路径
    fov_path = './21_training_mask.gif'  # 替换为实际的FOV掩码路径
    save_path = './results/result.png'  # 替换为实际的保存路径

    # 运行测试
    result = test_single_image(img_path, fov_path, net, save_path)