import os
import cv2
import numpy as np
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils

def _debug_image(image, mask_radius, coordinate, coordinate_data, save_path):
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
    file_name = os.path.join(save_path, (utils.GlobalConfig.DEVICE_ID +'_DC_ROI.png'))
    cv2.imwrite(file_name, rgb)    

def split_rect(roi):
    rows, cols = roi.shape
    cy , cx = rows // 2, cols // 2
    ul = roi[: cy, : cx].mean()
    ur = roi[: cy, cx:].mean()
    ll = roi[cy :, : cx].mean()
    lr = roi[cy :, cx:].mean()
    return ul, ur, ll, lr

def _dark_corner(image, center_x, center_y, roi_size, mask_radius, border_distance, image_size):
    theta = np.arctan(center_y / center_x)
    delta_y = round((mask_radius - border_distance) * np.sin(theta))  # 这里向下取整，防止顶点超出掩膜
    delta_x = round((mask_radius - border_distance) * np.cos(theta))
    roi_height = roi_size[0] * 2
    coordinate = []
    # roi_UL
    roi_UL = image[center_y - delta_y: center_y - delta_y + roi_height, center_x - delta_x: center_x - delta_x + roi_height]  # [449: 481, 449: 481]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x - delta_x, center_x - delta_x + roi_height])
    ul, ur, ll, lr = split_rect(roi_UL)
    dc_ul = (3 * ul) / (ur + ll + lr)

    # roi_UR
    roi_UR = image[center_y - delta_y: center_y - delta_y + roi_height, center_x + delta_x - roi_height: center_x + delta_x]  # [449: 481, 2543: 2575]
    coordinate.append([center_y - delta_y, center_y - delta_y + roi_height, center_x + delta_x - roi_height, center_x + delta_x])
    ul, ur, ll, lr = split_rect(roi_UR)
    dc_ur = (3 * ur) / (ul + ll + lr)


    # roi_LL
    roi_LL = image[center_y + delta_y - roi_height: center_y + delta_y, center_x - delta_x: center_x - delta_x + roi_height] # [2543: 2575, 449: 481] 
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x - delta_x, center_x - delta_x + roi_height])
    ul, ur, ll, lr = split_rect(roi_LL)
    dc_ll = (3 * ll) / (ul + ur + lr)


    # roi_LR
    roi_LR = image[center_y + delta_y - roi_height: center_y + delta_y, center_x + delta_x - roi_height: center_x + delta_x] # [2543: 2575, 2543: 2575]
    coordinate.append([center_y + delta_y - roi_height, center_y + delta_y, center_x + delta_x - roi_height, center_x + delta_x])
    ul, ur, ll, lr = split_rect(roi_LR)
    dc_lr = (3 * lr) / (ul + ur + ll)

    return dc_ul, dc_ur, dc_ll, dc_lr, coordinate

def dark_corner(image, roi_size, mask_radius, border_distance, csv_output, debug_flag=False, save_path=None):
    image_size = image.shape
    center_x =  image_size[1] // 2
    center_y =  image_size[0] // 2
    ul, ur, ll, lr, coordinate = _dark_corner(image, center_x, center_y, roi_size, mask_radius, border_distance, image_size)
    dc_data = {
                'DC_UL': str(100 * ul),
                'DC_UR': str(100 * ur),
                'DC_LL': str(100 * ll),
                'DC_LR': str(100 * lr)
            }
    if debug_flag or csv_output:
        
        os.makedirs(save_path, exist_ok=True)
    
    if debug_flag:
        data = [ul, ur, ll, lr]
        _debug_image(image, mask_radius, coordinate, data, save_path)
        
    if csv_output:
        save_file_path = os.path.join(save_path, 'dc_data.csv')
        utils.save_dict_to_csv(dc_data, str(save_file_path)) 


def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    dc_cfg = cfg.dark_corner
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if dc_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    # r, gr, gb, b = utils.split_channel(image, image_cfg.bayer_pattern)
    # half_roi_size = [ri_cfg.roi_size[0] // 2, ri_cfg.roi_size[1] // 2]
    # half_snr_roi_size = [ri_cfg.snr_roi_size[0] // 2, ri_cfg.snr_roi_size[1] // 2]
    # relative_illumination(r, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(gr, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(gb, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    # relative_illumination(b, half_roi_size, half_snr_roi_size, ri_cfg.mask_radius//2, ri_cfg.border_distance, ri_cfg.csv_output, ri_cfg.debug_flag, save_path)
    
    if dc_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)

    dark_corner(image, dc_cfg.roi_size, dc_cfg.mask_radius, dc_cfg.border_distance, dc_cfg.csv_output, dc_cfg.debug_flag, save_path)
    
    return 

if __name__ == '__main__':
    file_name = r'D:\Code\CameraTest\image\CV\light\Ketron_P0C_FF2_Line1_Light1_EOL-Light__030703111601010e0b0300001a08_20241229041233_0.raw'
    save_path = r'D:\Code\CameraTest\result'
    config_path = r'D:\Code\CameraTest\Config\config_cv.yaml'
    # utils.process_files(file_name, func, '.raw', save_path, config_path)
    func(file_name, save_path, config_path)
    
    print('dc finished!')
    
    
        
    





    