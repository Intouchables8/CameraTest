from pathlib import Path
import os
import numpy as np
from dotenv import load_dotenv
import cv2
from types import SimpleNamespace
# 加载env
load_dotenv()
DEFAULT_LOG_PATH = 'log.log'
LOG_PATH = os.getenv("LOG_PATH", DEFAULT_LOG_PATH)

def dict_to_obj(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_obj(i) for i in d]
    else:
        return d
     
def read_rgb(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    device_id = file_name.stem
    return image, device_id

# read_raw8():
    
def read_raw10(file_name, image_size):
    raw = np.fromfile(file_name, np.uint16)
    image = raw[:image_size[0] * image_size[1]].reshape(image_size)
    device_id = file_name.stem
    return image, device_id
    
def read_mipi10(file_name, image_size):
    rows, cols = image_size
    width = (cols * 10) // 8
    height = rows
    mipi_data = np.fromfile(file_name, dtype=np.uint8)
    device_id = file_name.stem
    
    mipi_data = mipi_data.reshape(height, -1)
    mipi_data = mipi_data[:, :width]
    
    raw_data = np.zeros(rows * cols, np.uint16)
    raw_data[0::4] = mipi_data[:, 0::5].ravel()
    raw_data[1::4] = mipi_data[:, 1::5].ravel()
    raw_data[2::4] = mipi_data[:, 2::5].ravel()
    raw_data[3::4] = mipi_data[:, 3::5].ravel()
    
    rawImageLow2bits = mipi_data[:, 4::5].ravel()
    raw_data <<= 2 # 高位左移 2 位
    
    raw_data[0::4] |= (rawImageLow2bits >> 0) & 3
    raw_data[1::4] |= (rawImageLow2bits >> 2) & 3
    raw_data[2::4] |= (rawImageLow2bits >> 4) & 3
    raw_data[3::4] |= (rawImageLow2bits >> 6) & 3
    
    image = raw_data.reshape(rows, cols)
    return image, device_id

def show_image_support(image, scale_factor=1, name='test'):
    image_size = image.shape
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, (int(image_size[1] // scale_factor), int(image_size[0] // scale_factor)))
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()