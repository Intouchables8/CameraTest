from pathlib import Path
import cv2
import numpy as np
import sys
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils

def _debug_image(image, mask_radius, coordinate, coordinate_data, save_path):
    utils.FilePath.create_folder(save_path)
    image_size = image.shape
    rgb = utils.raw_2_rgb(image)
    i = 0
    cv2.circle(rgb, (image_size[1] // 2, image_size[0] // 2), mask_radius, (0, 255, 0), 1)
    for coor in coordinate:
        rgb = cv2.rectangle(rgb, (coor[2], coor[0]), (coor[3], coor[1]), (0,255,255), thickness=2)    
        if coor[2] > image_size[1] - 100:
            coor[2] = image_size[1] - 100
        rgb = cv2.putText(rgb, str(np.round(coordinate_data[i],4)),  (coor[2], coor[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        i += 1
    file_name = save_path / (utils.GlobalConfig.DEVICE_ID +'_RI_ROI.png')
    cv2.imwrite(file_name, rgb)    

def _locate_roi(image, center_x, center_y, roi_size, mask_radius, border_distance, image_size):
    #   ----------------- U -----------------
    #   | UL                             UR |
    #   |                                   |
    #   L                 C                 R
    #   |                                   |
    #   | LL                             LR |
    #   ----------------- D -----------------
    theta = np.arctan(center_y / center_x)
    delta_y = round((mask_radius - border_distance) * np.sin(theta))  # 这里向下取整，防止顶点超出掩膜
    delta_x = round((mask_radius - border_distance) * np.cos(theta))
    roi_height = roi_size[0]
    half_roi_height = roi_height // 2

    # roi_U
    roi_U = image[border_distance: roi_height + border_distance, center_x - half_roi_height: center_x + half_roi_height]  # [8: 40, 1496: 1528]
    coordinate = [[border_distance, roi_height + border_distance, center_x - half_roi_height, center_x + half_roi_height]]
    mean_u = roi_U.mean()

    # roi_D
    roi_D = image[-roi_height - border_distance: image_size[0] -border_distance, center_x - half_roi_height: center_x + half_roi_height]  # [-40: -8, 1496: 1528]
    coordinate.append([image_size[0] -roi_height - border_distance, image_size[0] -border_distance, center_x - half_roi_height, center_x + half_roi_height])
    mean_d = roi_D.mean()

    # roi_L
    roi_L = image[center_y - half_roi_height: center_y + half_roi_height, border_distance: roi_height + border_distance] # [1496: 1528, 8 : 40]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, border_distance, roi_height + border_distance])
    mean_l = roi_L.mean() 

    # roi_R
    roi_R = image[center_y - half_roi_height: center_y + half_roi_height, -roi_height - border_distance: image_size[1] - border_distance] # [1496: 1528, -40: -8]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, image_size[1] -roi_height - border_distance, image_size[1] - border_distance])
    mean_r = roi_R.mean()

    # roi_C
    roi_C = np.round(image[center_y - half_roi_height: center_y + half_roi_height, center_x - half_roi_height: center_x + half_roi_height])  # [1496: 1528, 1496: 1528]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, center_x - half_roi_height, center_x + half_roi_height])
    mean_c = roi_C.mean()

    # roi_UL
    roi_UL = image[center_y - delta_y: center_y - delta_y + roi_height, center_x - delta_x: center_x - delta_x + roi_height]  # [449: 481, 449: 481]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x - delta_x, center_x - delta_x + roi_height])
    mean_ul = roi_UL.mean()

    # roi_UR
    roi_UR = image[center_y - delta_y: center_y - delta_y + roi_height, center_x + delta_x - roi_height: center_x + delta_x]  # [449: 481, 2543: 2575]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x + delta_x - roi_height, center_x + delta_x])
    mean_ur = roi_UR.mean()

    # roi_LL
    roi_LL = image[center_y + delta_y - roi_height: center_y + delta_y, center_x - delta_x: center_x - delta_x + roi_height] # [2543: 2575, 449: 481] 
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x - delta_x, center_x - delta_x + roi_height])
    mean_ll = roi_LL.mean()

    # roi_LR
    roi_LR = image[center_y + delta_y - roi_height: center_y + delta_y, center_x + delta_x - roi_height: center_x + delta_x] # [2543: 2575, 2543: 2575]
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x + delta_x - roi_height, center_x + delta_x])
    mean_lr = roi_LR.mean()
    return mean_u, mean_ul, mean_ur, mean_l, mean_c, mean_r, mean_ll, mean_lr, mean_d, coordinate

def _snr(image, center_x, center_y, roi_size, mask_radius, border_distance, image_size):
    #   ----------------- U -----------------
    #   | UL                             UR |
    #   |                                   |
    #   L                 C                 R
    #   |                                   |
    #   | LL                             LR |
    #   ----------------- D -----------------
    theta = np.arctan(center_y / center_x)
    delta_y = round((mask_radius - border_distance) * np.sin(theta))  # 这里向下取整，防止顶点超出掩膜
    delta_x = round((mask_radius - border_distance) * np.cos(theta))
    roi_height = roi_size[0]
    half_roi_height = roi_height // 2

    # roi_C
    roi_C = np.round(image[center_y - half_roi_height: center_y + half_roi_height, center_x - half_roi_height: center_x + half_roi_height])  # [1496: 1528, 1496: 1528]
    snr_c = roi_C.mean() / roi_C.std(ddof=1)

    # roi_UL
    roi_UL = image[center_y - delta_y: center_y - delta_y + roi_height, center_x - delta_x: center_x - delta_x + roi_height]  # [449: 481, 449: 481]
    snr_ul = roi_UL.mean() / roi_UL.std(ddof=1)

    # roi_UR
    roi_UR = image[center_y - delta_y: center_y - delta_y + roi_height, center_x + delta_x - roi_height: center_x + delta_x]  # [449: 481, 2543: 2575]
    snr_ur = roi_UR.mean() / roi_UR.std(ddof=1)

    # roi_LL
    roi_LL = image[center_y + delta_y - roi_height: center_y + delta_y, center_x - delta_x: center_x - delta_x + roi_height] # [2543: 2575, 449: 481] 
    snr_ll = roi_LL.mean() / roi_LL.std(ddof=1)

    # roi_LR
    roi_LR = image[center_y + delta_y - roi_height: center_y + delta_y, center_x + delta_x - roi_height: center_x + delta_x] # [2543: 2575, 2543: 2575]
    snr_lr = roi_LR.mean() / roi_LR.std(ddof=1)
    return snr_c, snr_ul, snr_ur, snr_ll, snr_lr

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
    
    
        
    





    