import numpy as np
import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(str(ROOTPATH))
from Common import utils
import os

#region
# 速度提升百分之50，但是初次运行需要编译
# @njit
# def compute_optical_center(image, gray_thresh):
#     """使用 `@njit` 计算光学中心"""
#     image_size = image.shape
#     height, width = image_size

#     # ✅ 用 `for` 循环代替 `image[image <= gray_thresh] = 0`
#     for y in range(height):
#         for x in range(width):
#             if image[y, x] <= gray_thresh:
#                 image[y, x] = 0

#     low = np.sum(image)

#     # ✅ 直接生成 `x_u` 和 `y_u`
#     x_u = np.zeros((height, width), dtype=np.float64)
#     y_u = np.zeros((height, width), dtype=np.float64)

#     for y in range(height):
#         for x in range(width):
#             x_u[y, x] = (x + 0.5) * image[y, x]
#             y_u[y, x] = (y + 0.5) * image[y, x]

#     x_u = np.sum(x_u)
#     y_u = np.sum(y_u)

#     oc_x = x_u / low
#     oc_y = y_u / low
#     offset_x = oc_x - width / 2
#     offset_y = oc_y - height / 2

#     return oc_x, oc_y, offset_x, offset_y

# def optical_center(src_image, thresh, csv_output, save_path):
#     """预处理 `np.bincount()` 计算灰度阈值"""
#     image = src_image.copy().astype(np.float64)  # ✅ 确保 `Numba` 兼容
#     total_pixel = image.size
#     pixel_thresh = thresh * total_pixel

#     # ✅ 用 `np.bincount()` 计算像素直方图（比 `np.unique()` 更快）
#     counts = np.bincount(image.ravel().astype(np.int32), minlength=256)
#     cumulative_counts = np.cumsum(counts)

#     # ✅ 用 `np.searchsorted()` 找到灰度阈值（比 `argwhere()` 快）
#     gray_thresh = np.searchsorted(cumulative_counts, pixel_thresh)

#     # ✅ `@njit` 计算光学中心
#     return compute_optical_center(image, gray_thresh)
#endregion
def optical_center(src_image, thresh=0.9, csv_output=False, save_path=None):
    image = src_image.copy()
    image_size = image.shape
    total_pixel = image.size
    pxiel_thresh = thresh * total_pixel
    gray_val, counts = np.unique(image, return_counts=True) # 像素值和对应的数量
    index = np.argwhere(np.cumsum(counts) >= pxiel_thresh)[0]
    gray_thresh = gray_val[index]  # 找到百分之90数量的灰度阈值
    image[image <= gray_thresh] = 0  # 只保留图像满足阈值的部分进行计算
    
    low = image.sum()
    x_u = np.arange(0.5, image_size[1], 1,dtype='float64')
    x_u = np.expand_dims(x_u, axis=0).repeat(image_size[0], axis=0) # 将行数据重复480列
    x_u = (x_u * image).sum()
    y_u = np.arange(0.5, image_size[0],1, dtype='float64')
    y_u = np.expand_dims(y_u, axis=1).repeat(image_size[1], axis=1) 
    y_u = (y_u * image).sum()
    
    oc_x =  x_u / low
    oc_y =  y_u / low
    offset_x = oc_x - image_size[1] / 2 
    offset_y = oc_y - image_size[0] / 2
    data = {
                    'OC_x': str(oc_x),
                    'OC_y': str(oc_y),
                    'Offset_x': str(offset_x),
                    'Offset_y': str(offset_y),
    }
    if csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'oc_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))
    return data

def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    oc_cfg = cfg.optical_center
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if oc_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    if oc_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)
    
    optical_center(image, oc_cfg.thresh, oc_cfg.csv_output, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\RGB\light.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_rgb.yaml'
    func(file_name, save_path, config_path)
    print('OC finished!')