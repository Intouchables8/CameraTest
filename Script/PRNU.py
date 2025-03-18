from pathlib import Path
import cv2
import numpy as np
import sys
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils

def prnu(images, roi_size, csv_output, debug_flag=False, save_path=None):
    save_path = Path(save_path)
    p_l = images.mean(axis=2)  
    rows, cols = p_l.shape
    half_roi_height = roi_size[0] // 2
    cy, cx = rows // 2, cols // 2
    center_avg = p_l[cy - half_roi_height: cy + half_roi_height, cx - half_roi_height: cx + half_roi_height].mean()         
    # 每个像素在所有帧上的均值
    # dark_noise_fpn_total: 每个像素均值与总体均值（P_total/(L*N)）之间的总体标准差
    fpn_total = np.std(p_l, ddof=0)  # 这里用总体标准差

    # fpn_row: 计算每行均值的相邻差分（循环处理），并归一化
    h_n = p_l.mean(axis=1)
    fpn_row = np.sqrt(np.mean(np.square((h_n - np.roll(h_n, 1)) / np.sqrt(2))))

    # fpn_col: 同理，计算每列均值的相邻差分（循环处理）
    v_n = p_l.mean(axis=0)
    fpn_col = np.sqrt(np.mean(np.square((v_n - np.roll(v_n, 1)) / np.sqrt(2))))

    # fpn_pixel: 根据总 FPN 和行、列分量求解像素级 FPN
    fpn_pixel = np.sqrt(fpn_total**2 - fpn_row**2 - fpn_col**2)
    
    prnu_total = fpn_total / center_avg
    prnu_row = fpn_row / center_avg
    prnu_col = fpn_col / center_avg
    prnu_pixel = fpn_pixel / center_avg

    data = {
                'PRNU_Pixel': str(prnu_pixel),
                'PRNU_Row': str(prnu_row),
                'PRNU_Col': str(prnu_col),
                'PRNU_Total': str(prnu_total),
            }
    if csv_output:
        save_path = save_path / 'PRNU_data.csv'
        utils.save_dict_to_csv(data, save_path) 


def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    prnu_cfg = cfg.PRNU
    images = utils.load_images(file_name, prnu_cfg.image_count,image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if prnu_cfg.sub_black_level:
        images = utils.sub_black_level(images, image_cfg.black_level)

    prnu(images, prnu_cfg.roi_size, prnu_cfg.csv_output, prnu_cfg.debug_flag, save_path)
    
    return 

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\CV\light\Ketron_P0C_FF2_Line1_Light1_EOL-Light__030703111601010e0b0300001a08_20241229041233_0.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_cv.yaml'
    # utils.process_file_or_folder(file_name, '.raw', func, save_path, config_path)
    func(file_name, save_path, config_path)
    
    print('PRNU finished!')
    
    
        
    





    