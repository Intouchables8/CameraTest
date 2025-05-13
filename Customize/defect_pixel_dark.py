import numpy as np
import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils
import os
import cv2

def _debug_image(image, ind_dpd, avg_image, diff_image, channel_name, save_path):
    specify = False
    channel_data =[]
    x_data =[]
    y_data =[]
    value_data =[]
    avg_data =[]
    diff_data =[]
    name = ['channel', 'x', 'y', 'value', 'image mean', 'diff']
    for i in range(len(ind_dpd[0])):
        channel_data.append(channel_name)
        x_data.append(str(ind_dpd[1][i]))
        y_data.append(str(ind_dpd[0][i]))
        value_data.append(image[ind_dpd[0][i], ind_dpd[1][i]])
        avg_data.append(avg_image)
        diff_data.append(diff_image[ind_dpd[0][i], ind_dpd[1][i]])

    data = zip(channel_data, x_data, y_data, value_data, avg_data, diff_data)
    save_file_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_dpd.csv')) 
    utils.save_lists_to_csv(data, name, str(save_file_path))
    
#region
    # 对比某个图像输出的数据对应的坐标  
    # file = r''
    # if specify:
    #     import pandas as pd
    #     df = pd.read_csv(file)
        # save_file_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_Compare.csv') 
    #     if channel_name == 'B':
    #         filtered_df = df[df['Channel'] == 'B']
    #         com_defect_pixel = [filtered_df['Y'].tolist(), filtered_df['X'].tolist()]
    #     if channel_name == 'Gb':
    #         filtered_df = df[df['Channel'] == 'Gb']
    #         com_defect_pixel = [filtered_df['Y'].tolist(), filtered_df['X'].tolist()]
    #     if channel_name == 'Gr':
    #         filtered_df = df[df['Channel'] == 'Gr']
    #         com_defect_pixel = [filtered_df['Y'].tolist(), filtered_df['X'].tolist()]
    #     if channel_name == 'R':
    #         filtered_df = df[df['Channel'] == 'R']
    #         com_defect_pixel = [filtered_df['Y'].tolist(), filtered_df['X'].tolist()]
    #     writeLog(com_defect_pixel, save_file_path)
#endregion

def _dpd_support(image, channel_name, thresh, debug_flag, save_path, distribution):
    total_dpd = 0
    singlet_dpd = 0
    doublet_dpdt = 0
    triplet_dpd = 0
    image_size = image.shape
    avg_image = image.mean()
    diff_image = np.abs(image - avg_image)
    
    if distribution:
        above_50 = np.count_nonzero(diff_image > 50)
        above_100 = np.count_nonzero(diff_image > 100)
        above_150 = np.count_nonzero(diff_image > 150)
        above_200 = np.count_nonzero(diff_image > 200)
        above = [above_50, above_100, above_150, above_200]
    else:
        above = [0, 0, 0, 0]
        
    ind_dpd = np.where(diff_image > thresh)
    
    # 找到single double triple的数量
    dpd_map = np.zeros(image_size, np.uint8)
    if len(ind_dpd[0]) > 0:
        total_dpd = len(ind_dpd[0])
        dpd_map[ind_dpd[0], ind_dpd[1]] = 255
        num, _, area, _ = cv2.connectedComponentsWithStats(dpd_map, connectivity=8)
        index_single = []
        index_double = []
        index_triple = []
        for i in range(num):
            if area[i,4] == 1:
                singlet_dpd = singlet_dpd + 1
                index_single.append(i)
            if area[i,4] == 2:
                doublet_dpdt = doublet_dpdt + 1
                index_double.append(i)
            if area[i,4] >= 3 and area[i,4] < 10000:
                triplet_dpd = triplet_dpd + 1
                index_triple.append(i)
        
        if debug_flag:
            if len(ind_dpd[0]) > 0:
                
                os.makedirs(save_path, exist_ok=True)
                _debug_image(image, ind_dpd, avg_image, diff_image, channel_name, save_path)
        
        
    return total_dpd, singlet_dpd, doublet_dpdt, triplet_dpd, above, dpd_map

def defect_pixel_dark(image, bayer_pattern, thresh, csv_output, debug_flag=False, save_path=None, distribution=False):
    if bayer_pattern == 'RGGB' or bayer_pattern == 'BGGR':
        r, gr, gb, b = utils.split_channel(image, bayer_pattern)
        r_total_dpd, r_singlet_dpd, r_doublet_dpd, r_triplet_dpd, above, _ = _dpd_support(r, 'R', thresh, debug_flag, save_path, distribution)
        gr_total_dpd, gr_singlet_dpd, gr_doublet_dpd, gr_triplet_dpd, above, _ = _dpd_support(gr, 'Gr', thresh, debug_flag, save_path, distribution)
        gb_total_dpd, gb_singlet_dpd, gb_doublet_dpd, gb_triplet_dpd, above, _ = _dpd_support(gb, 'Gb', thresh, debug_flag, save_path, distribution)
        b_total_dpd, b_singlet_dpd, b_doublet_dpd, b_triplet_dpd, above, _ = _dpd_support(b, 'B', thresh, debug_flag, save_path, distribution)
        y_total_dpd, y_singlet_dpd, y_doublet_dpd, y_triplet_dpd, above, dpd_map = _dpd_support(image, 'orgin', thresh, debug_flag, save_path, distribution)
        data = { 
                    'DPD_Contrast_Threshold': str(thresh),

                    'DPD_Singlet_count_R': str(r_singlet_dpd),
                    'DPD_Doublet_count_R': str(r_doublet_dpd),
                    'DPD_Triplet_count_R': str(r_triplet_dpd),
                    'DPD_R_Total_Area': str(r_total_dpd),
                    
                    'DPD_Singlet_count_Gb': str(gb_singlet_dpd),
                    'DPD_Doublet_count_Gb': str(gb_doublet_dpd),
                    'DPD_Triplet_count_Gb': str(gb_triplet_dpd),
                    'DPD_Gb_Total_Area': str(gb_total_dpd),
                    
                    'DPD_Singlet_count_Gr': str(gr_singlet_dpd),
                    'DPD_Doublet_count_Gr': str(gr_doublet_dpd),
                    'DPD_Triplet_count_Gr': str(gr_triplet_dpd),
                    'DPD_Gr_Total_Area': str(gr_total_dpd),
                    
                    'DPD_Singlet_count_B': str(b_singlet_dpd),
                    'DPD_Doublet_count_B': str(b_doublet_dpd),
                    'DPD_Triplet_count_B': str(b_triplet_dpd),
                    'DPD_B_Total_Area': str(b_total_dpd),
                    
                    'DPD_Singlet_count_G': str(gb_singlet_dpd + gr_singlet_dpd),
                    'DPD_Doublet_count_G': str(gr_doublet_dpd + gb_doublet_dpd),
                    'DPD_Triplet_count_G': str(gr_triplet_dpd + gb_triplet_dpd),
                    'DPD_G_Total_Area': str(gr_total_dpd + gb_total_dpd),
                    
                    'DPD_Singlet_RBGrGbSum_Ct': str(b_singlet_dpd + gb_singlet_dpd + gr_singlet_dpd + r_singlet_dpd),
                    'DPD_Doublet_RBGrGbSum_Ct': str(b_doublet_dpd + gb_doublet_dpd + gr_doublet_dpd + r_doublet_dpd),
                    'DPD_Triplet_RBGrGbSum_Ct': str(b_triplet_dpd + gb_triplet_dpd + gr_triplet_dpd + r_triplet_dpd),
                    'DPD_Total_Area_RBGrGbSum': str(b_total_dpd + gb_total_dpd + gr_total_dpd + r_total_dpd),
                    
                    'DPD_Singlet_count': str(y_singlet_dpd),
                    'DPD_Doublet_count': str(y_doublet_dpd),
                    'DPD_Triplet_count': str(y_triplet_dpd),
                    'DPD_Total_Area': str(y_total_dpd),  
        }
    else:
        total_dpd, singlet_dpd, doublet_dpd, triplet_dpd, above, dpd_map = _dpd_support(image, 'orgin', thresh, debug_flag, save_path, distribution)
        above_50, above_100, above_150, above_200 = above 
        data = {
            'dpd Contrast Threshold': str(thresh),
            'dpd_Singlet_count': str(singlet_dpd),
            'dpd_Doublet_count': str(doublet_dpd),
            'dpd_Triplet_count': str(triplet_dpd),
            'dpd_Total_count': str(total_dpd),
            'Above_50': str(above_50),
            'Above_100': str(above_100),
            'Above_150': str(above_150),
            'Above_200': str(above_200),
        }
    
    if csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'dpd_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))
    return data, dpd_map

def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).dark
    image_cfg = cfg.image_info
    dpd_cfg = cfg.defect_pixel_dark
    if dpd_cfg.image_count > 1:
        images = utils.load_images(file_name, dpd_cfg.image_count, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)
        image = images.mean(axis=2)
    else:
        image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)
        
    if dpd_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    defect_pixel_dark(image, dpd_cfg.input_pattern, dpd_cfg.thresh, dpd_cfg.csv_output, dpd_cfg.debug_flag, save_path, dpd_cfg.distribution)
    return True

if __name__ == '__main__':
    file_name = r'C:\Users\wangjianan\Desktop\Innorev_Result\DPD\image'
    save_path = r'C:\Users\wangjianan\Desktop\Innorev_Result\DPD'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    utils.process_files(file_name, func, '.raw',save_path, config_path)
    print('dpd finished!')