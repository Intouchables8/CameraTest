import numpy as np
import cv2
from pathlib import Path
import sys
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils
from concurrent.futures import ThreadPoolExecutor

def _debug_image(image, ind_dpl, avg_image, diff_image, diff_image_pre, channel_name, save_path):
    specify = False
    utils.FilePath.create_folder(save_path)
    save_file_path = save_path / (utils.GlobalConfig.get_device_id() + '_DPL.csv') 
    channel_data =[]
    x_data =[]
    y_data =[]
    value_data =[]
    avg_data =[]
    diff_data =[]
    diff_pre_data =[]
    name = ['channel', 'x', 'y', 'value', 'avg', 'diff', 'diff_precent']
    for i in range(len(ind_dpl[0])):
        channel_data.append(channel_name)
        x_data.append(str(ind_dpl[1][i]))
        y_data.append(str(ind_dpl[0][i]))
        value_data.append(image[ind_dpl[0][i], ind_dpl[1][i]])
        avg_data.append(avg_image[ind_dpl[0][i], ind_dpl[1][i]])
        diff_data.append(diff_image[ind_dpl[0][i], ind_dpl[1][i]])
        diff_pre_data.append(diff_image_pre[ind_dpl[0][i], ind_dpl[1][i]])

    data = zip(channel_data, x_data, y_data, value_data, avg_data, diff_data, diff_pre_data)
    utils.save_lists_to_csv(data, name, save_file_path)
    # 对比某个图像输出的数据对应的坐标  
    # file = r''
    # if specify:
    #     import pandas as pd
    #     df = pd.read_csv(file)
        # save_file_path = save_path / (utils.GlobalConfig.get_device_id() + '_Compare.csv') 
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

def _dpl_support(src, channel_name, roi_size, thresh, mask_radius, debug_flag, save_path):
    image = src.copy()
    total_dpl = 0
    singlet_dpl = 0
    doublet_dplt = 0
    triplet_dpl = 0
    image_size = image.shape
    mask = utils.generate_mask(image_size, mask_radius)
    image[mask <=0] = 0
    kern = np.ones(roi_size)
    mask[mask > 0] = 1
    mask_conv = cv2.filter2D(mask.astype(np.uint16), -1, kern, borderType=cv2.BORDER_CONSTANT)  # mask 31*31 范围内像素数量
    mask_conv[mask < 1] = 0  # 图像加掩膜
    masked_image_conv = cv2.filter2D(image.astype(np.float64), -1, kern, borderType=cv2.BORDER_CONSTANT)
    masked_image_conv[mask < 1] = 0
    avg_image = np.zeros(image_size)
    avg_image[mask >= 1] = masked_image_conv[mask >= 1] / mask_conv[mask >= 1]  # 31*31区域灰度值的均值
    diff_image = np.abs(image - avg_image)
    diff_image_pre = np.zeros(image_size)
    diff_image_pre[mask >= 1] = diff_image[mask >= 1] / avg_image[mask >= 1]
    ind_dpl = np.where(diff_image_pre > thresh)
    
    # 找到single double triple的数量
    DPLMap = np.zeros(image_size, np.uint8)
    if len(ind_dpl[0]) > 0:
        total_dpl = len(ind_dpl[0])
        DPLMap[ind_dpl[0], ind_dpl[1]] = 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(DPLMap, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # 或直接 stats[1:, 4]
            singlet_dpl = int((areas == 1).sum())
            doublet_dplt = int((areas == 2).sum())
            triplet_dpl = int((areas >= 3).sum())
        
        if debug_flag:
            if len(ind_dpl[0]) > 0:
                _debug_image(image, ind_dpl, avg_image, diff_image, diff_image_pre, channel_name, save_path) 
    return total_dpl, singlet_dpl, doublet_dplt, triplet_dpl

def defect_pixel_light(image, bayer_pattern, roi_size, mask_radius, thresh, csv_output, debug_flag=False, save_path=None, multi_thread=True):
    save_path = Path(save_path)
    if bayer_pattern == 'RGGB' or bayer_pattern == 'BGGR':
        y_image = utils.bayer_2_y(image, bayer_pattern)
        r, gr, gb, b = utils.split_channel(image, bayer_pattern)
        half_roi_size = (roi_size[0] // 2, roi_size[0] // 2)
        half_mask_radius = mask_radius // 2
        
        if not multi_thread:
            r_total_dpl, r_singlet_dpl, r_doublet_dpl, r_triplet_dpl = _dpl_support(r, 'R',half_roi_size, thresh, half_mask_radius, debug_flag, save_path)
            gr_total_dpl, gr_singlet_dpl, gr_doublet_dpl, gr_triplet_dpl = _dpl_support(gr, 'Gr',half_roi_size, thresh, half_mask_radius, debug_flag, save_path)
            gb_total_dpl, gb_singlet_dpl, gb_doublet_dpl, gb_triplet_dpl = _dpl_support(gb, 'Gb',half_roi_size, thresh, half_mask_radius, debug_flag, save_path)
            b_total_dpl, b_singlet_dpl, b_doublet_dpl, b_triplet_dpl = _dpl_support(b, 'B',half_roi_size, thresh, half_mask_radius, debug_flag, save_path)
            y_total_dpl, y_singlet_dpl, y_doublet_dpl, y_triplet_dpl = _dpl_support(y_image, 'y', roi_size, thresh, mask_radius, debug_flag, save_path)
        else:
            tasks = [
                (r, 'R', half_roi_size, thresh, half_mask_radius, debug_flag, save_path),
                (gr, 'Gr', half_roi_size, thresh, half_mask_radius, debug_flag, save_path),
                (gb, 'Gb', half_roi_size, thresh, half_mask_radius, debug_flag, save_path),
                (b, 'B', half_roi_size, thresh, half_mask_radius, debug_flag, save_path),
                (y_image, 'y', roi_size, thresh, mask_radius, debug_flag, save_path)
            ]
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_dpl_support, *params) for params in tasks]
                results = [future.result() for future in futures]
            r_total_dpl, r_singlet_dpl, r_doublet_dpl, r_triplet_dpl = results[0]
            gr_total_dpl, gr_singlet_dpl, gr_doublet_dpl, gr_triplet_dpl = results[1]
            gb_total_dpl, gb_singlet_dpl, gb_doublet_dpl, gb_triplet_dpl = results[2]
            b_total_dpl, b_singlet_dpl, b_doublet_dpl, b_triplet_dpl = results[3]
            y_total_dpl, y_singlet_dpl, y_doublet_dpl, y_triplet_dpl = results[4]
        
 
        dpl_data = {
                    'DPL_Contrast_Threshold': str(100 * thresh),
                    'DPL_Singlet_count': str(y_singlet_dpl),
                    'DPL_Singlet_RBGrGbSum_Ct count': str(b_singlet_dpl + gb_singlet_dpl + gr_singlet_dpl + r_singlet_dpl),
                    'DPL_Singlet_count_R': str(r_singlet_dpl),
                    'DPL_Singlet_count_Gb': str(gb_singlet_dpl),
                    'DPL_Singlet_count_Gr': str(gr_singlet_dpl),
                    'DPL_Singlet_count_G': str(gb_singlet_dpl + gr_singlet_dpl),
                    'DPL_Singlet_count_B': str(b_singlet_dpl),
                    
                    'DPL_Doublet_count': str(y_doublet_dpl),
                    'DPL_Doublet_RBGrGbSum_Ct': str(b_doublet_dpl + gb_doublet_dpl + gr_doublet_dpl + r_doublet_dpl),
                    'DPL_Doublet_count_R': str(r_doublet_dpl),
                    'DPL_Doublet_count_Gb': str(gb_doublet_dpl),
                    'DPL_Doublet_count_Gr': str(gr_doublet_dpl),
                    'DPL_Doublet_count_G': str(gr_doublet_dpl + gb_doublet_dpl),
                    'DPL_Doublet_count_B': str(b_doublet_dpl),
                    
                    'DPL_Triplet_count': str(y_triplet_dpl),
                    'DPL_Triplet_RBGrGbSum_Ct': str(b_triplet_dpl + gb_triplet_dpl + gr_triplet_dpl + r_triplet_dpl),
                    'DPL_Triplet_count_R': str(r_triplet_dpl),
                    'DPL_Triplet_count_Gb': str(gb_triplet_dpl),
                    'DPL_Triplet_count_Gr': str(gr_triplet_dpl),
                    'DPL_Triplet_count_G': str(gr_triplet_dpl + gb_triplet_dpl),
                    'DPL_Triplet_count_B': str(b_triplet_dpl),
                    
                    'DPL_R_Total_Area': str(r_total_dpl),
                    'DPL_Gr_Total_Area': str(gr_total_dpl),
                    'DPL_Gb_Total_Area': str(gb_total_dpl),
                    'DPL_G_Total_Area': str(gr_total_dpl + gb_total_dpl),
                    'DPL_B_Total_Area': str(b_total_dpl),
                    'DPL_Total_Area': str(y_total_dpl),
                    'DPL_Total_Area_RBGrGbSum': str(b_total_dpl + gb_total_dpl + gr_total_dpl + r_total_dpl)
        }
    else:
        total_dpl, singlet_dpl, doublet_dpl, triplet_dpl = _dpl_support(image, 'Y', roi_size, thresh, mask_radius, debug_flag, save_path)
        dpl_data = {
            'DPL Contrast Threshold': str(100 * thresh),
            'DPL_Singlet_count': str(singlet_dpl),
            'DPL_Doublet_count': str(doublet_dpl),
            'DPL_Triplet_count': str(triplet_dpl),
            'DPL_Total_count': str(total_dpl)
        }
    if csv_output:
        save_file_path = save_path / 'dpl_data.csv'
        utils.save_dict_to_csv(dpl_data, save_file_path)

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    dpl_cfg = cfg.defect_pixel_light
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if dpl_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
        
    if dpl_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)
        
    defect_pixel_light(image, dpl_cfg.input_pattern, dpl_cfg.roi_size, dpl_cfg.mask_radius, dpl_cfg.thresh, dpl_cfg.csv_output,dpl_cfg.debug_flag, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\california\Light.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_california.yaml'
    func(file_name, save_path, config_path)
    
    print('dpl finished!')
