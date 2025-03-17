import sys
from pathlib import Path
common_path = Path(__file__).parent
sys.path.append(str(common_path))
from utils_support import *
import logging
import math
import numpy as np
import cv2
from typing import Literal
from datetime import datetime
import csv
import yaml
from numba import njit
import time
from functools import wraps

class GlobalConfig:
    __DEVICE_ID = 'DEFAULT'
    @classmethod
    def set_device_id(cls, device_id):
        cls.DEVICE_ID = device_id
    @classmethod
    
    def get_device_id(cls):
        return cls.DEVICE_ID

def load_config(file_path):
    if not file_path.exists():
        raise KeyError(f'load_config: unable find config path: {file_path}')
        
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return dict_to_obj(config)

def draw_mark_with_text(image, x, y, text, mark_radius=5, mark_color=(0, 255, 255),  text_offset_x=10, text_offset_y=0, text_size=1, text_color=(0, 168, 255), text_thickness=2):
    x = int(round(x))
    y = int(round(y))
    cv2.circle(image, (x, y), radius=mark_radius, color=mark_color, thickness=-1)
    cv2.putText(image, str(text), (x + text_offset_x, y + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

def draw_rect_with_text(image, x, y, w, h, text, rect_color=(0, 255, 0), rect_thickness=2, 
                        text_offset_x=10, text_offset_y=30, 
                        text_size=1.3, text_color=(0, 100, 255), text_thickness=2):
    cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, rect_thickness)
    cv2.putText(image, str(text), (x + text_offset_x, y + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

def calcu_distance(points_xy, center_xy):
    distance = np.sqrt(np.sum((points_xy - center_xy)**2, axis=1))
    return distance

def sort_order_index(points_xy, center_xy, delta_angle=15, clockwise=False):
    '''
        1 -------- 2
        |          |
        |          |
        4 -------- 3 
    '''
    delta_rad = np.deg2rad(delta_angle)
    if clockwise:
        angles = np.arctan2(-(points_xy[:,1] - center_xy[1]), (points_xy[:, 0] - center_xy[0])) + delta_rad
    else:
        angles = np.arctan2(-(points_xy[:,1] - center_xy[1]), (-(points_xy[:, 0] - center_xy[0]))) + delta_rad
    angles = angles % (2 * np.pi)  # 将弧度从-pi~pi转换为0~2pi
    index = np.argsort(angles)
    return index
    
def group_data(data, thresh):
    data = np.array(data)
    sorted_indices = np.argsort(data)  
    sorted_data = data[sorted_indices]  # 按照排序索引排序数据
    diffs = np.diff(sorted_data)
    boundaries = np.where(diffs > thresh)[0]
    grouped_data = np.split(sorted_data, boundaries + 1)
    grouped_indices = np.split(sorted_indices, boundaries + 1)  # 按相同的边界分割索引
    return grouped_data, grouped_indices
    
def log_message(level: Literal['INFO', 'WARNING', 'ERROR'], message: str, log_path: str = LOG_PATH):
    logging.basicConfig(
        level=logging.DEBUG,  # 记录所有级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path,
        filemode="a"  # 追加模式
    )

    # 获取 logger
    logger = logging.getLogger(__name__)
    level = level.upper()
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        raise ValueError(f"Unsupported log level: {level}")

    # 也输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    
    # 避免重复添加 handler
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
              
def save_dict_to_csv(data, save_path):
    # 检查文件是否存在
    file_exists = save_path.exists()

    # 获取当前时间并格式化
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 打开 CSV 文件，若文件存在则追加，若文件不存在则创建并写入数据
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # 如果文件不存在，则写入第一行：包含所有的 key 和时间戳
        if not file_exists:
            writer.writerow(['Device ID'] + ['Data Time']+ list(data.keys()))  # 写入所有的 key，作为第一行
            # 将时间戳附加到最后一个值后面
            writer.writerow([GlobalConfig.get_device_id()] + [current_time] + list(data.values()))  # 写入所有的 value + 时间戳，作为第二行
        else:
            # 如果文件已存在，追加新的值和时间戳
            writer.writerow([GlobalConfig.get_device_id()] + [current_time] + list(data.values())) 
     
def save_lists_to_csv(data, name, save_path):
    file_exists = save_path.exists()
    data = list(data)
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(name)
        if isinstance(data, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in data):
            writer.writerows(data)  # 处理单行数据
        else:
            writer.writerow(data)  # 处理嵌套列表（多行数据）
     
def sub_black_level(image, black_level):
    image = np.clip(image, black_level, None)
    image = image - black_level
    return image

def bayer_2_y(bayer_image, bayer_pattern):
    if bayer_pattern == 'BGGR':
        mode = cv2.COLOR_BayerBGGR2RGB_EA
    elif bayer_pattern == 'RGGB':
        mode = cv2.COLOR_BayerRGGB2RGB_EA
    else:
        log_message('ERROR', f'bayer_2_y: unsupport {bayer_pattern}')
        raise TypeError('ERROR', f'bayer_2_y: unsupport {bayer_pattern}')
    rgb = cv2.cvtColor(bayer_image, mode)
    # RGB 图像转YUV的 Y 通道 -- >  Y = 0.2126 R + 0.7152 G + 0.0722 B   
    y_image = np.round(0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.uint16)
    return y_image

def search_files(file_path, extension, recursive=False):
    path = Path(file_path)
    if path.is_file():
        dir = path.parent
    else:
        dir = path

    if recursive:
        files = list(dir.rglob(f'*{extension}'))  # 递归查找
    else:
        files = list(dir.glob(f'*{extension}'))  # 仅当前目录
    return files

def crop_image(image, crop_tblr):
    image_size = image.shape
    if image.ndim == 2:
        crop_image = image[crop_tblr[0]: image_size[0] - crop_tblr[1], crop_tblr[2]: image_size[1] - crop_tblr[3]]
    else:
        image.ndim == 3
        crop_image = image[:, crop_tblr[0]: image_size[0] - crop_tblr[1], crop_tblr[2]: image_size[1] - crop_tblr[3]]
    return crop_image
    
def load_image(file_name: str, image_type: Literal['RGB', 'RAW8', 'RAW10', 'MIPI10'], image_size, crop_tblr= [0,0,0,0]):
    if image_type == 'RGB':
        image, device_id = read_rgb(file_name)
    elif image_type == 'RAW8':
        print('raw8 is not written')
        exit()
    elif image_type == 'RAW10':
        image, device_id = read_raw10(file_name, image_size)
    elif image_type == 'MIPI10':
        print('raw8 is not written')
        exit()
    else:
        log_message('ERROR', f'load_image: unsupport {image_type}')
        raise TypeError('ERROR', f'load_image: unsupport {image_type}')
    
    if any(x !=0 for x in crop_tblr):
        image = crop_image(image, crop_tblr)
    GlobalConfig.set_device_id(device_id)
    return image

def load_images(file_name, file_count, image_type: Literal['RGB', 'RAW8', 'RAW10', 'MIPI10'], image_size, crop_tblr= [0,0,0,0]):
    file_path = Path(file_name)
    extension = file_path.suffix
    files_name = search_files(file_name, extension)
    cnt = len(files_name)
    if cnt != file_count:
        raise ValueError(f"load_images: only find {cnt} files, {file_count} needed!")
    images = np.zeros((image_size[0], image_size[1], file_count))
    for i, name in enumerate(files_name):
        images[:, :, i] = load_image(name, image_type, image_size)
    if any(x !=0 for x in crop_tblr):
        images = crop_image(images, crop_tblr)
    return images

def generate_mask(image_size, mask_radius):
    mask = np.zeros(image_size, np.uint8)
    center = (image_size[1] // 2, image_size[0] // 2)
    cv2.circle(mask, center, mask_radius, 255, -1)
    return mask

def generate_circle(image_size, center_xy, raidus_cur, delta_angle, roi_size, border_distance=0):
    half_roi_height = roi_size[0] // 2
    raidus_cur = int(raidus_cur)
    angle = np.arange(0, 360, delta_angle)
    x_center = np.round(center_xy[0] + raidus_cur * np.round(np.cos(np.radians(angle)), 4)).astype(int)  # 每个框中心位置的x值,四舍五入，取整
    y_center = np.round(center_xy[1] + raidus_cur * np.sin(np.radians(angle))).astype(int)  # 每个框中心位置的y值，四舍五入，取整
    index = np.where((y_center - 1 >= half_roi_height + border_distance) & 
                     (y_center - 1 <= image_size[0] - half_roi_height - border_distance)&
                     (x_center - 1 >= half_roi_height + border_distance) & 
                     (x_center - 1 <= image_size[1] - half_roi_height - border_distance))  # 限制roi框的边缘不超界
    x_center = x_center[index]  # 满足条件不会超界的框
    y_center = y_center[index]  # 满足条件不会超界的框
    roi_center_xy = [(x, y) for x, y in zip(x_center, y_center)] # 将坐标变为二位点数组
    return roi_center_xy

def generate_roi(center_xy, roi_size):
    center_xy = np.array(center_xy).reshape(-1,2)
    ts = center_xy.shape
    roi_size = np.repeat(roi_size, ts[0], axis=0).reshape(2, ts[0]).T
    half_size = roi_size // 2
    num_mod = roi_size % 2
    #if size is odd like :15 --> 1 2 3 4 5 6 7 cxy 7 6 5 4 3 2 1
    #if size is even like:14 --> 1 2 3 4 5 6 7 cxy 6 5 4 3 2 1
    dst_roi = np.append(center_xy - half_size, center_xy + half_size + num_mod,axis=1)
    dst_roi = dst_roi.astype(np.int32)
    return dst_roi

def split_channel(image, bayter_pattern):
    tl = image[0 :: 2, 0 :: 2]
    tr = image[0 :: 2, 1 :: 2]
    bl = image[1 :: 2, 0 :: 2]
    br = image[1 :: 2, 1 :: 2]
    if bayter_pattern == 'RGGB':
        r = tl
        gr = tr
        gb = bl
        b = br
    if bayter_pattern == 'BGGR':
        b = tl
        gb = tr
        gr = bl
        r = br
    return r, gr, gb, b
    
def draw_circle(image, mask_radius, color=(255, 0, 0), thickness=1):
    image_size = image.shape
    center = (image_size[1] // 2, image_size[0] // 2)
    cv2.circle(image, center, mask_radius, color, thickness)
    return image

def raw_2_rgb(image):
    if image.dtype == np.uint16:
        image = (image / 4).astype(np.uint8)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return rgb
    
def show_image(image, scale_factor=1, name='test'):
    if image.dtype == np.uint16:
        image_8bit = (image >> 2).astype(np.uint8)
        show_image_support(image_8bit, scale_factor, name)
    elif image.dtype == np.uint8:
        show_image_support(image, scale_factor, name)
    else:
        range = image.max() - image.min()
        norm_image = (image - image.min()) / range
        show_image_support(norm_image, scale_factor, name)

class FilePath:
    @staticmethod
    def create_folder(file_name):
        path = Path(file_name)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_save_folder_name(file_name, folder_name):
        path = Path(file_name)
        current_directory = path.parent if path.is_file() else path
        new_folder_path = current_directory / folder_name
        return new_folder_path
        
@njit
def calcu_ciede2000(Lab_1, Lab_2):
    # 预计算常量
    C_25_7 = 6103515625.0  # 25**7
    two_pi = 2 * math.pi
    pi_6 = math.pi / 6.0
    pi_30 = math.pi / 30.0
    deg63 = 63 * math.pi / 180.0

    # 解包 Lab 值
    L1 = Lab_1[0]
    a1 = Lab_1[1]
    b1 = Lab_1[2]
    L2 = Lab_2[0]
    a2 = Lab_2[1]
    b2 = Lab_2[2]

    # 初始计算色度
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_ave_init = (C1 + C2) / 2.0
    # 计算 G 因子，缓存 C_ave_init**7
    C_ave_init_pow7 = C_ave_init ** 7
    G = 0.5 * (1 - math.sqrt(C_ave_init_pow7 / (C_ave_init_pow7 + C_25_7)))

    # 计算调整后的 a 值，不变的 L 和 b
    a1_ = (1 + G) * a1
    a2_ = (1 + G) * a2
    L1_, L2_ = L1, L2
    b1_, b2_ = b1, b2

    # 计算新的色度
    C1_ = math.sqrt(a1_ * a1_ + b1_ * b1_)
    C2_ = math.sqrt(a2_ * a2_ + b2_ * b2_)

    # 计算色相角 h
    if a1_ == 0.0 and b1_ == 0.0:
        h1_ = 0.0
    else:
        h1_ = math.atan2(b1_, a1_)
        if h1_ < 0:
            h1_ += two_pi

    if a2_ == 0.0 and b2_ == 0.0:
        h2_ = 0.0
    else:
        h2_ = math.atan2(b2_, a2_)
        if h2_ < 0:
            h2_ += two_pi

    # 计算亮度差、色度差
    dL_ = L2_ - L1_
    dC_ = C2_ - C1_

    # 计算色相差
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0.0:
        dh_ = 0.0
    else:
        if dh_ > math.pi:
            dh_ -= two_pi
        elif dh_ < -math.pi:
            dh_ += two_pi
    dH_ = 2.0 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2.0)

    # 计算平均亮度和平均色度（使用调整后的色度）
    L_ave = (L1_ + L2_) / 2.0
    C_ave_final = (C1_ + C2_) / 2.0

    # 计算平均色相
    dh_abs = abs(h1_ - h2_)
    sh_sum = h1_ + h2_
    if C1_ * C2_ == 0.0:
        h_ave = h1_ + h2_
    else:
        if dh_abs <= math.pi:
            h_ave = (h1_ + h2_) / 2.0
        else:
            if sh_sum < two_pi:
                h_ave = (h1_ + h2_) / 2.0 + math.pi
            else:
                h_ave = (h1_ + h2_) / 2.0 - math.pi

    # 计算 T 因子（使用预计算常量）
    T = (1 - 0.17 * math.cos(h_ave - pi_6)
         + 0.24 * math.cos(2.0 * h_ave)
         + 0.32 * math.cos(3.0 * h_ave + pi_30)
         - 0.2 * math.cos(4.0 * h_ave - deg63))

    # 将 h_ave 转换为角度并归一化
    h_ave_deg = h_ave * 180.0 / math.pi
    if h_ave_deg < 0.0:
        h_ave_deg += 360.0
    elif h_ave_deg > 360.0:
        h_ave_deg -= 360.0

    # 计算 dTheta
    dTheta = 30.0 * math.exp(-(((h_ave_deg - 275.0) / 25.0) ** 2))

    # 缓存 C_ave_final**7，用于后续计算 R_C
    C_ave_final_pow7 = C_ave_final ** 7
    R_C = 2.0 * math.sqrt(C_ave_final_pow7 / (C_ave_final_pow7 + C_25_7))
    S_C = 1 + 0.045 * C_ave_final
    S_H = 1 + 0.015 * C_ave_final * T

    Lm50s = (L_ave - 50.0) ** 2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20.0 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90.0) * R_C

    # 归一化亮度、色度和色相差
    f_L = dL_ / S_L
    f_C = dC_ / S_C
    f_H = dH_ / S_H

    dE_00 = math.sqrt(f_L * f_L + f_C * f_C + f_H * f_H + R_T * f_C * f_H)
    dC_00 = math.sqrt(f_C * f_C + f_H * f_H + R_T * f_C * f_H)
    return dE_00, dC_00

def process_file_or_folder(input_path, extension, func, *args, **kwargs):
    """
    通用方法：判断 input_path 是文件还是文件夹，并执行 func 处理。
    
    :param input_path: 文件或文件夹路径
    :param func: 需要执行的处理函数（可自定义参数）
    :param *args: 额外的 *args 参数，会传递给 func
    :param **kwargs: 额外的 **kwargs 参数，会传递给 func
    """
    input_path = Path(input_path)
    if input_path.is_file():
        func(input_path, *args, **kwargs)
    
    elif input_path.is_dir():  # 检查是否是目录
        for file_path in input_path.iterdir():  # 遍历目录下的所有文件/文件夹
            if file_path.is_file() and file_path.suffix == extension:  # 检查是否是指定后缀的文件
                func(file_path, *args, **kwargs) 
    
class time_block:
    """用于测量代码块执行时间的上下文管理器"""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.start_time
        print(f"📌 {self.name} 执行时间: {elapsed_time:.6f} 秒")

def time_it_avg(repeats=3):
    """装饰器：重复执行多次，输出每次执行时间，并记录函数内部调用的时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for i in range(repeats):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)  # 运行被装饰函数
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                times.append(elapsed_time)
                print(f"⏳ 第 {i+1} 次执行 `{func.__name__}`: {elapsed_time:.6f} 秒")

            avg_time = sum(times) / repeats
            print(f"✅ 函数 `{func.__name__}` 平均执行时间 ({repeats} 次): {avg_time:.6f} 秒")
            return result  # 返回原始函数的执行结果
        return wrapper
    return decorator
