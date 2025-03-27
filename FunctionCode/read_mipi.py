import cv2
import numpy as np
import os
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).parent
sys.path.append(str(ROOTPATH))
from Common import utils

def read_mipi(file_name, rows, cols, save_path, output_type, crop_tblr):
    width = (cols * 10) // 8
    height = rows
    
    mipi_data = np.fromfile(file_name, dtype=np.uint8)
    device_id = os.path.splitext(os.path.basename(file_name))[0]
    
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
    
    
    
    os.makedirs(save_path, exist_ok=True)
    if 'png' in output_type:

        save_file_path = os.path.join(save_path, (device_id + '.png'))
        image = (image >> 2).astype(np.uint8)
        cv2.imwrite(save_file_path, image)
    else:
        image = utils.crop_image(image, crop_tblr)
        save_file_path = os.path.join(save_path, (device_id + '.raw'))
        image.tofile(save_file_path)
    

def mipi_2_type(file_name, rows, cols, save_path, output_type, crop_tblr):
    
    if path.is_file():
        read_mipi(file_name, rows, cols, save_path, output_type)
    elif path.is_dir():
        # for file in path.rglob('*'):  # 递归遍历所有文件
        for file in path.glob('*'):  # 遍历所有文件
            if file.is_file():
                read_mipi(file, rows, cols, save_path, output_type, crop_tblr)

if __name__ == '__main__':
    file_name = r'E:\Wrok\Temp\CaliforniaFATP\20250312\20250311\offline\holder_1\356YW33GB6001T\20250311171830\Dark\noise\camera'     
    output_type = '.raw'        
    cols = 3072                 
    rows = 3024      
    crop_tblr = [0, 0, 0, 48]

    save_path = utils.FilePath.generate_save_folder_name(file_name, 'result')
    mipi_2_type(file_name, rows, cols, save_path, output_type, crop_tblr)
