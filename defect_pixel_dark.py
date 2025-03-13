import numpy as np
from Common import utils
from pathlib import Path
import cv2

def _debug_image(image, ind_dpd, avg_image, diff_image, channel_name, save_path):
    specify = False
    utils.FilePath.create_folder(save_path)
    save_file_path = save_path / (utils.GlobalConfig.get_device_id() + '_dpd.csv') 
    channel_data =[]
    x_data =[]
    y_data =[]
    value_data =[]
    avg_data =[]
    diff_data =[]

    name = ['channel', 'x', 'y', 'value', 'avg', 'diff']
    for i in range(len(ind_dpd[0])):
        channel_data.append(channel_name)
        x_data.append(str(ind_dpd[1][i]))
        y_data.append(str(ind_dpd[0][i]))
        value_data.append(image[ind_dpd[0][i], ind_dpd[1][i]])
        avg_data.append(avg_image[ind_dpd[0][i], ind_dpd[1][i]])
        diff_data.append(diff_image[ind_dpd[0][i], ind_dpd[1][i]])

    data = zip(channel_data, x_data, y_data, value_data, avg_data, diff_data)
    utils.save_lists_to_csv(data, name, save_file_path)
    
#region
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
#endregion

def _CalDSNU(image, roi_size):
    # ## --------------------------------------
    # #  2024 04 03 DPD的位置替换成图像均值  by Sheree And Alex
    # avg = image.mean()
    # image[defect_pixel_map > 0] = avg
    # ##----------------------------------------
    integralImage = cv2.integral(image)  # 对图像的每个点与之前的所有点进行求和
    avgBlockArray = np.zeros((189, 189), np.float64)
    avgBlocks = []
    W, H = image.shape[1], image.shape[0]
    col = 0
    roi_size_height = roi_size[0]
    for x in range(0, W, roi_size_height):
        row = 0
        for y in range(0, H, roi_size_height):
            br = (min(y + roi_size_height, H), min(x + roi_size_height, W))  # 右下角
            tl = (y, x)  # 左上角
            tr = (y, min(x + roi_size_height, W))  # 右上角
            bl = (min(y + roi_size_height, H), x)  # 左下角
            sumBlock = integralImage[br] - integralImage[tr] - integralImage[bl] + integralImage[tl]
            avgBlock = sumBlock / ((br[0] - tl[0]) * (br[1] - tl[1]))
            avgBlocks.append(avgBlock)
            avgBlockArray[row, col] = avgBlock
            row += 1
        col += 1
    DSNU = max(avgBlocks) - min(avgBlocks)
    return DSNU, max(avgBlocks), min(avgBlocks)

def _dpd_support(image, channel_name, thresh, debug_flag, save_path):
    total_dpd = 0
    singlet_dpd = 0
    doublet_dpdt = 0
    triplet_dpd = 0
    image_size = image.shape
    avg_image = image.mean()
    diff_image = np.abs(image - avg_image)
    ind_dpd = np.where(diff_image > thresh)
    
    # 找到single double triple的数量
    dpdMap = np.zeros(image_size, np.uint8)
    if len(ind_dpd[0]) > 0:
        total_dpd = len(ind_dpd[0])
        dpdMap[ind_dpd[0], ind_dpd[1]] = 255
        num, _, area, _ = cv2.connectedComponentsWithStats(dpdMap, connectivity=8)
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
            _debug_image(image, ind_dpd, avg_image, diff_image, channel_name, save_path)
        
    return total_dpd, singlet_dpd, doublet_dpdt, triplet_dpd

def defect_pixel_dark(image, bayer_pattern, roi_size, thresh, csv_output, debug_flag=False, save_path=None):
    save_path = Path(save_path)
    if bayer_pattern == 'RGGB' or bayer_pattern == 'BGGR':
        r, gr, gb, b = utils.split_channel(image, bayer_pattern)
        r_total_dpd, r_singlet_dpd, r_doublet_dpd, r_triplet_dpd = _dpd_support(r, 'R', thresh, debug_flag, save_path)
        gr_total_dpd, gr_singlet_dpd, gr_doublet_dpd, gr_triplet_dpd = _dpd_support(gr, 'Gr', thresh, debug_flag, save_path)
        gb_total_dpd, gb_singlet_dpd, gb_doublet_dpd, gb_triplet_dpd = _dpd_support(gb, 'Gb', thresh, debug_flag, save_path)
        b_total_dpd, b_singlet_dpd, b_doublet_dpd, b_triplet_dpd = _dpd_support(b, 'B', thresh, debug_flag, save_path)
        y_total_dpd, y_singlet_dpd, y_doublet_dpd, y_triplet_dpd = _dpd_support(image, 'orgin', thresh, debug_flag, save_path)
        dpd_data = { 
                    'DPD_Contrast_Threshold': str(thresh),
                    'DPD_Singlet_count': str(y_singlet_dpd),
                    'DPD_Singlet_RBGrGbSum_Ct count': str(b_singlet_dpd + gb_singlet_dpd + gr_singlet_dpd + r_singlet_dpd),
                    'DPD_Singlet_count_R': str(r_singlet_dpd),
                    'DPD_Singlet_count_Gb': str(gb_singlet_dpd),
                    'DPD_Singlet_count_Gr': str(gr_singlet_dpd),
                    'DPD_Singlet_count_G': str(gb_singlet_dpd + gr_singlet_dpd),
                    'DPD_Singlet_count_B': str(b_singlet_dpd),
                    
                    'DPD_Doublet_count': str(y_doublet_dpd),
                    'DPD_Doublet_RBGrGbSum_Ct': str(b_doublet_dpd + gb_doublet_dpd + gr_doublet_dpd + r_doublet_dpd),
                    'DPD_Doublet_count_R': str(r_doublet_dpd),
                    'DPD_Doublet_count_Gb': str(gb_doublet_dpd),
                    'DPD_Doublet_count_Gr': str(gr_doublet_dpd),
                    'DPD_Doublet_count_G': str(gr_doublet_dpd + gb_doublet_dpd),
                    'DPD_Doublet_count_B': str(b_doublet_dpd),
                    
                    'DPD_Triplet_count': str(y_triplet_dpd),
                    'DPD_Triplet_RBGrGbSum_Ct': str(b_triplet_dpd + gb_triplet_dpd + gr_triplet_dpd + r_triplet_dpd),
                    'DPD_Triplet_count_R': str(r_triplet_dpd),
                    'DPD_Triplet_count_Gb': str(gb_triplet_dpd),
                    'DPD_Triplet_count_Gr': str(gr_triplet_dpd),
                    'DPD_Triplet_count_G': str(gr_triplet_dpd + gb_triplet_dpd),
                    'DPD_Triplet_count_B': str(b_triplet_dpd),
                    
                    'DPD_R_Total_Area': str(r_total_dpd),
                    'DPD_Gr_Total_Area': str(gr_total_dpd),
                    'DPD_Gb_Total_Area': str(gb_total_dpd),
                    'DPD_G_Total_Area': str(gr_total_dpd + gb_total_dpd),
                    'DPD_B_Total_Area': str(b_total_dpd),
                    'DPD_Total_Area': str(y_total_dpd),
                    'DPD_Total_Area_RBGrGbSum': str(b_total_dpd + gb_total_dpd + gr_total_dpd + r_total_dpd)
        }
    else:
        total_dpd, singlet_dpd, doublet_dpd, triplet_dpd = _dpd_support(image, 'orgin', thresh, debug_flag, save_path)
        dpd_data = {
            'dpd Contrast Threshold': str(thresh),
            'dpd_Singlet_count': str(singlet_dpd),
            'dpd_Doublet_count': str(doublet_dpd),
            'dpd_Triplet_count': str(triplet_dpd),
            'dpd_Total_count': str(total_dpd)
        }
    
    # 计算DSNU
    DSNU, max_dsnu, min_dsnu = _CalDSNU(image, roi_size)
    dpd_data['DSNU'] = str(DSNU)
    dpd_data['Max_DSNU'] = str(max_dsnu)
    dpd_data['MIN_DSNU'] = str(min_dsnu)
    
    if csv_output:
        save_file_path = save_path / 'dpd_data.csv'
        
        utils.save_dict_to_csv(dpd_data, save_file_path)

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
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
    
    defect_pixel_dark(image, dpd_cfg.input_pattern, dpd_cfg.roi_size, dpd_cfg.thresh, dpd_cfg.csv_output, dpd_cfg.debug_flag, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\Script\image\Dark\dpd\California_P0_DARK_1_2_Dark16X_352RK1AFBV00K5_3660681a28230823610100_20231226_122724_0.raw'
    save_path = r'G:\Script\result'
    config_path = r'G:\Script\Config\config.yaml'
    import time 
    start = time.time()
    func(file_name, save_path, config_path)
    print(time.time() - start)
    print('dpd finished!')