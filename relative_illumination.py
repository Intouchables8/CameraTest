from pathlib import Path
import cv2
import numpy as np
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
    mean_U = roi_U.mean()

    # roi_D
    roi_D = image[-roi_height - border_distance: image_size[0] -border_distance, center_x - half_roi_height: center_x + half_roi_height]  # [-40: -8, 1496: 1528]
    coordinate.append([image_size[0] -roi_height - border_distance, image_size[0] -border_distance, center_x - half_roi_height, center_x + half_roi_height])
    mean_D = roi_D.mean()

    # roi_L
    roi_L = image[center_y - half_roi_height: center_y + half_roi_height, border_distance: roi_height + border_distance] # [1496: 1528, 8 : 40]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, border_distance, roi_height + border_distance])
    mean_L = roi_L.mean() 

    # roi_R
    roi_R = image[center_y - half_roi_height: center_y + half_roi_height, -roi_height - border_distance: image_size[1] - border_distance] # [1496: 1528, -40: -8]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, image_size[1] -roi_height - border_distance, image_size[1] - border_distance])
    mean_R = roi_R.mean()

    # roi_C
    roi_C = np.round(image[center_y - half_roi_height: center_y + half_roi_height, center_x - half_roi_height: center_x + half_roi_height])  # [1496: 1528, 1496: 1528]
    coordinate.append([center_y - half_roi_height, center_y + half_roi_height, center_x - half_roi_height, center_x + half_roi_height])
    mean_C = roi_C.mean()

    # roi_UL
    roi_UL = image[center_y - delta_y: center_y - delta_y + roi_height, center_x - delta_x: center_x - delta_x + roi_height]  # [449: 481, 449: 481]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x - delta_x, center_x - delta_x + roi_height])
    mean_UL = roi_UL.mean()

    # roi_UR
    roi_UR = image[center_y - delta_y: center_y - delta_y + roi_height, center_x + delta_x - roi_height: center_x + delta_x]  # [449: 481, 2543: 2575]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x + delta_x - roi_height, center_x + delta_x])
    mean_UR = roi_UR.mean()

    # roi_LL
    roi_LL = image[center_y + delta_y - roi_height: center_y + delta_y, center_x - delta_x: center_x - delta_x + roi_height] # [2543: 2575, 449: 481] 
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x - delta_x, center_x - delta_x + roi_height])
    mean_LL = roi_LL.mean()

    # roi_LR
    roi_LR = image[center_y + delta_y - roi_height: center_y + delta_y, center_x + delta_x - roi_height: center_x + delta_x] # [2543: 2575, 2543: 2575]
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x + delta_x - roi_height, center_x + delta_x])
    mean_LR = roi_LR.mean()
    return mean_U, mean_UL, mean_UR, mean_L, mean_C, mean_R, mean_LL, mean_LR, mean_D, coordinate

def relative_illumination(image, roi_size, snr_roi_size, mask_radius, border_distance, csv_output, debug_flag=False, save_path=None):
    save_path = Path(save_path)
    image_size = image.shape
    center_x =  int(image_size[1]/2)
    center_y =  int(image_size[0]/2)
    mean_U, mean_UL, mean_UR, mean_L, mean_C, mean_R, mean_LL, mean_LR, mean_D, coordinate = _locate_roi(image, center_x, center_y, roi_size, mask_radius, border_distance, image_size)
    ri_U, ri_UL, ri_UR, ri_L, ri_R, ri_LL, ri_LR, ri_D = np.array([mean_U, mean_UL, mean_UR, mean_L, mean_R, mean_LL, mean_LR, mean_D]) / mean_C
    ri_corner = np.array([ri_UL, ri_UR, ri_LL, ri_LR])
    ri_udrl = np.array([ri_U, ri_D, ri_L, ri_R])
    ri_corner_min = ri_corner.min()
    ri_corner_delta_max = ri_corner.max() - ri_corner.min()
    ri_durl_min = ri_udrl.min()
    ri_durl_delta_max = ri_udrl.max() - ri_udrl.min()
    coordinate_data = [ri_U, ri_D, ri_L, ri_R, mean_C, ri_UL, ri_UR, ri_LL, ri_LR]
    ri_Center = mean_C
    # SNR
    half_snr_roi_height = snr_roi_size[0] // 2
    snr_roi = image[center_y - half_snr_roi_height: center_y+ half_snr_roi_height, center_x -half_snr_roi_height: center_x + half_snr_roi_height]
    snr_mean = snr_roi.mean()
    snr_std = snr_roi.std(ddof=1)
    snr = snr_mean / snr_std

    if debug_flag:
        _debug_image(image, mask_radius, coordinate, coordinate_data, save_path)
        
    ri_data = {
                'RI_Corner_Min': str(100 * ri_corner_min),
                'RI_UDLR_Min': str(100 * ri_durl_min),
                'RI_Corner_Delta_Max': str(100 * ri_corner_delta_max),
                'RI_UDLR_Delta_Max': str(100 * ri_durl_delta_max),
                'RI_Center_Mean_LSB': str(ri_Center),
                'RI_U': str(100 * ri_U),
                'RI_D': str(100 * ri_D),
                'RI_L': str(100 * ri_L),
                'RI_R': str(100 * ri_R),
                'RI_UL': str(100 * ri_UL),
                'RI_UR': str(100 * ri_UR),
                'RI_LL': str(100 * ri_LL),
                'RI_LR': str(100 * ri_LR),
                'SNR': str(snr)
            }
    if csv_output:
        save_path = save_path / 'ri_data.csv'
        utils.save_dict_to_csv(ri_data, save_path) 
    return True

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    ri_cfg = cfg.relative_illumination
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if ri_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    # r, gr, gb, b = utils.split_channel(image, image_cfg.bayer_pattern)
    # half_roi_size = [ri_cfg.roi_size[0] // 2, ri_cfg.roi_size[1] // 2]
    # half_snr_roi_size = [ri_cfg.snr_roi_size[0] // 2, ri_cfg.snr_roi_size[1] // 2]
    # relative_illumination(r, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(gr, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(gb, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(b, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    
    if ri_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)

        relative_illumination(image, ri_cfg.roi_size, ri_cfg.snr_roi_size, ri_cfg.mask_radius, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    
    
    return True

if __name__ == '__main__':
    file_name = r'E:\Wrok\Temp\Oregon\0304\AH4 verification\test'
    save_path = r'E:\Wrok\Temp\Oregon\0304\AH4 verification\test'
    config_path = r'G:\Script\Config\test.yaml'
    utils.process_file_or_folder(file_name, '.raw', func, save_path, config_path)
    # func(file_name, save_path, config_path)
    
    print('RI finished!')
    
    
        
    





    