import numpy as np
import cv2
import os
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils
from Common import ciede2000
from Common.utils import *

def _debug_image(image, rgb, mask_radius, num_Ring, all_roi_rect, val_ciede, index_max_min, fov_rings, save_path):
    rgb = (rgb >> 2).astype(np.uint8)
    devece_id = utils.GlobalConfig.get_device_id()
    image_size = image.shape
    # mask = utils.generate_mask(image_size, mask_radius)
    # heat_map = (image >> 2)
    # for ring in all_roi_rect:
    #     for pos in ring:
    #         cv2.rectangle(heat_map, (pos[0], pos[1]), (pos[2], pos[3]), 0)
    # import matplotlib.pyplot as plt
    # plt.ioff() # 关闭交互模式 这样不显示
    # plt.imshow(heat_map, cmap='jet')
    # save_image_path = os.path.join(save_path, (devece_id + '_CU_HeatMap.png')
    # plt.savefig(save_image_path, dpi=450)
    # plt.close
    #  画圈  标记ROI  标记最大值最小值-------------------------------
    i = 0
    for ring in all_roi_rect:
        j = 0
        for pos in ring:
            if j == index_max_min[i, 0]:  # 最大值
                text = 'Max: ' + str(np.round(val_ciede[i, 0], 4))
                cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 255), 2)
                cv2.putText(rgb, text, (pos[0] - 30, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            elif j == index_max_min[i, 1]:  # 最大值
                text = 'Min: ' + str(val_ciede[i, 1])
                cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)
                cv2.putText(rgb, text, (pos[0] - 30, pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:    
                cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 255))
            j += 1
        i += 1 
    for i in range(num_Ring):
        x = 50
        y = 50 + 60 * i
        # 增加文字框背景
        background = np.zeros_like(rgb)
        background[y - 30: y + 10, x - 3: x + 700] = (0, 255, 255)
        rgb = cv2.addWeighted(rgb, 1, background, 0.5, 0)
        text = str(fov_rings[i]) + 'FOV' + ' Max:' + str(np.round(val_ciede[i, 0].max(), 3)) + ' Min:' + str(np.round(val_ciede[i, 0].min(), 3)) + ' Range:' + str(np.round(val_ciede[i, 0].max() - val_ciede[i, 1].min(), 3))
        cv2.putText(rgb, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    save_image_path = os.path.join(save_path, (devece_id + '_CU_image.png'))
    cv2.imwrite(save_image_path, rgb) 
    
def _delta_c(r_img, gr_img, gb_img, b_img, lab_full, lab1, image_height, image_width, roi_height, radius):
    y_cord = np.arange(0, image_height, roi_height)
    x_cord = np.arange(0, image_width, roi_height)
    center_x = image_width //2
    center_y = image_height // 2
    radius_sq = radius * radius

    # 初始化 delta_C0 和 delta_C1 的最大/最小值
    max_delta_c0 = 0.0
    min_delta_c0 = np.inf
    max_delta_c1 = 0.0
    min_delta_c1 = np.inf
    
    for x in x_cord:
        for y in y_cord:
            # 用平方比较避免 np.sqrt 计算
            dx = x - center_x  # x 对应列坐标，center_xy[1] 为中心列
            dy = y - center_y  # y 对应行坐标，center_xy[0] 为中心行
            if dx*dx + dy*dy > radius_sq:
                continue
            
            # 确保 ROI 在图像边界内
            x1, y1 = x, y
            x2, y2 = x + roi_height, y + roi_height
            if x2 > image_width or y2 > image_height:
                continue
            
            ## DeltaC 
            lab2 = lab_full[y1:y2, x1:x2, :]
            # 利用 numpy 向量化计算 ROI 内各通道均值
            # cur_lab2_arr = lab2.mean(axis=0).mean(axis=0)
            cur_lab2 = (lab2[:,:,0].mean(), lab2[:,:,1].mean(), lab2[:,:,2].mean())
            # _, delta_c0 = utils.calcu_ciede2000(lab1, cur_lab2)
            _, delta_c0 = ciede2000.calcu_ciede2000(lab1, cur_lab2)
            
            if delta_c0 > max_delta_c0:
                max_delta_c0 = delta_c0
            if delta_c0 < min_delta_c0:
                min_delta_c0 = delta_c0

            # --- 计算 R/G - B/G 相关的 DeltaC1 ---
            # 直接从预先拆分好的通道中取当前 ROI
            half_x1, half_x2 = x1 // 2, x2 // 2
            half_y1, half_y2 = y1 // 2, y2 // 2
            roi_r  = r_img[half_y1: half_y2, half_x1:half_x2]
            roi_gr = gr_img[half_y1: half_y2, half_x1:half_x2]
            roi_gb = gb_img[half_y1: half_y2, half_x1:half_x2]
            roi_b  = b_img[half_y1: half_y2, half_x1:half_x2]
            # 利用 numpy 的 mean 方法计算均值
            r_mean  = roi_r.mean()
            gr_mean = roi_gr.mean()
            gb_mean = roi_gb.mean()
            b_mean  = roi_b.mean()
            
            deltaC1 = np.abs((b_mean - r_mean) / gb_mean / 2 + (b_mean - r_mean) / gr_mean / 2)
            
            if deltaC1 > max_delta_c1:
                max_delta_c1 = deltaC1
            if deltaC1 < min_delta_c1:
                min_delta_c1 = deltaC1
    range_delta_C0 = max_delta_c0 - min_delta_c0
    range_delta_C1 = max_delta_c1 - min_delta_c1
    return range_delta_C0, range_delta_C1

# @time_it_avg(10)
def _old_cu(r_img, gr_img, gb_img, b_img, center_factor=8, roi_size=[20, 20]):
    rows, cols = r_img.shape
    cx, cy = cols // 2, rows // 2
    center_width, center_height = cols // center_factor, rows // center_factor
    center_width_half, center_height_half = center_width // 2, center_height // 2
    c_row_start = cy - center_height_half
    c_row_end = c_row_start + center_height
    c_col_start = cx - center_width_half 
    c_col_end = c_col_start + center_width 
    
    center_r = r_img[c_row_start: c_row_end, c_col_start: c_col_end]
    center_gr = gr_img[c_row_start: c_row_end, c_col_start: c_col_end]
    center_gb = gb_img[c_row_start: c_row_end, c_col_start: c_col_end]
    center_b = b_img[c_row_start: c_row_end, c_col_start: c_col_end]
    
    center_grgb = (center_gr + center_gb) * 0.5

    r_g = center_r / center_grgb
    b_g = center_b / center_grgb
    
    c1 = 1 / r_g.mean()
    c2 = 1 / b_g.mean()
    
    R = (r_img * c1).astype(np.float32)
    B = (b_img * c2).astype(np.float32)

    r_gr = R / gr_img
    b_gr = B / gr_img
    r_gb = R / gb_img
    b_gb = B / gb_img
    
    r_g2 = (r_gr + r_gb) * 0.5
    b_g2 = (b_gr + b_gb) * 0.5

    # anchor如何设置   后续需要进一步考虑
    r_g2 = cv2.blur(r_g2, roi_size, cv2.BORDER_CONSTANT)
    b_g2 = cv2.blur(b_g2, roi_size, cv2.BORDER_CONSTANT)
    
    diff = np.abs(r_g2 - b_g2)

    mx = np.max(diff)
    mx_y, mx_x = np.unravel_index(np.argmax(diff), diff.shape)
    CU_Ratio_RGr = r_gr[mx_y, mx_x]
    CU_Ratio_RGb = r_gb[mx_y, mx_x]
    CU_Ratio_BGr = b_gr[mx_y, mx_x]
    CU_Ratio_BGb = b_gb[mx_y, mx_x]
    return mx, CU_Ratio_RGr, CU_Ratio_RGb, CU_Ratio_BGr, CU_Ratio_BGb

    

def color_uniformity(image, bayer_pattern, roi_size, mask_radius, fov_rings, calcu_delta_c, calcu_old_cu, csv_output, debug_flag=False, save_path=None):
    image_size = image.shape
    if not image.dtype == np.uint16:
        image = (np.round(image)).astype(np.uint16)
    if bayer_pattern == 'BGGR':
        mode = cv2.COLOR_BayerBGGR2RGB_EA
    elif bayer_pattern == 'RGGB':
        mode = cv2.COLOR_BayerRGGB2RGB_EA
    else:
        utils.log_message('ERROR', f'color_uniformity: unsupport {bayer_pattern}')
        raise TypeError('ERROR', f'color_uniformity: unsupport {bayer_pattern}')

    rgb = cv2.cvtColor(image, mode)
    lab_full = cv2.cvtColor((rgb / 1023).astype('float32'), cv2.COLOR_RGB2Lab)
    center_xy = (image_size[1] // 2,image_size[0] // 2)
    center_rect = utils.generate_roi(center_xy, roi_size)
    center_roi = lab_full[center_rect[0, 1]: center_rect[0, 3], center_rect[0, 0]: center_rect[0, 2], :]
    lab_center_arr = center_roi.mean(axis=(0, 1))
    lab_center = (lab_center_arr[0], lab_center_arr[1], lab_center_arr[2])

    # 计算半径 ROI
    num_Ring = len(fov_rings)
    ringRadius = np.array(fov_rings) * mask_radius # 第一圈到第n圈
    angle = np.array([90, 45, 22.5, 22.5, 22.5])
    roi_center_xy = []
    for i in range(num_Ring):
        cur_angle = angle[i]
        cur_radius = ringRadius[i]
        # 生成中心点坐标
        center_roi_xy = utils.generate_circle(image_size, center_xy, cur_radius, cur_angle, roi_size,)
        roi_center_xy.append(center_roi_xy)  # 每一圈满足条件的中心点

    num_Fov = len(roi_center_xy)
    val_ciede = np.zeros((num_Fov, 5), np.float32)
    all_roi_rect = []
    index_max_min = np.zeros((num_Fov, 2),np.uint8)  # 最小值最大值索引
    
    for j in range(0,num_Fov):
        rect_roi = utils.generate_roi(roi_center_xy[j], roi_size)
        all_roi_rect.append(rect_roi)
        cur_num_roi = len(roi_center_xy[j])
        delta_E = np.zeros((cur_num_roi),np.float32)

        for i in range(0, cur_num_roi):
            lab2 = lab_full[rect_roi[i, 1]: rect_roi[i, 3], rect_roi[i, 0]: rect_roi[i, 2], :]
            cur_lab2 = (lab2[:, :, 0].mean(), lab2[:, :, 1].mean(), lab2[:, :, 2].mean())
            # delta_E[i], _ = utils.calcu_ciede2000(lab_center, cur_lab2)
            delta_E[i], _ = ciede2000.calcu_ciede2000(lab_center, cur_lab2)

        val_ciede[j,0] = delta_E.max()
        val_ciede[j,1] = delta_E.min()
        val_ciede[j,2] = delta_E.mean()
        val_ciede[j,3] = delta_E.std(ddof=1) 
        val_ciede[j,4] = val_ciede[j, 0] - val_ciede[j, 1]   # 0XF range
        index_max_min[j, 0] = delta_E.argmax()
        index_max_min[j, 1] = delta_E.argmin()
    

    data = {
                    'CU_01F_Max': str(val_ciede[0,0]),
                    'CU_01F_Min': str(val_ciede[0,1]),
                    'CU_01F_Mean': str(val_ciede[0,2]),
                    'CU_01F_Std': str(val_ciede[0,3]),
                    'CU_01F_Range': str(val_ciede[0,0] - val_ciede[0,1]),
                    'CU_03F_Max': str(val_ciede[1,0]),
                    'CU_03F_Min': str(val_ciede[1,1]),
                    'CU_03F_Mean': str(val_ciede[1,2]),
                    'CU_03F_Std': str(val_ciede[1,3]),
                    'CU_03F_Range': str(val_ciede[1,0] - val_ciede[1,1]),
                    'CU_05F_Max': str(val_ciede[2,0]),
                    'CU_05F_Min': str(val_ciede[2,1]),
                    'CU_05F_Mean': str(val_ciede[2,2]),
                    'CU_05F_Std': str(val_ciede[2,3]),
                    'CU_05F_Range': str(val_ciede[2,0] - val_ciede[2,1]),
                    'CU_07F_Max': str(val_ciede[3,0]),
                    'CU_07F_Min': str(val_ciede[3,1]),
                    'CU_07F_Mean': str(val_ciede[3,2]),
                    'CU_07F_Std': str(val_ciede[3,3]),
                    'CU_07F_Range': str(val_ciede[3,0] - val_ciede[3,1]),
                    'CU_09F_Max': str(val_ciede[4,0]),
                    'CU_09F_Min': str(val_ciede[4,1]),
                    'CU_09F_Mean': str(val_ciede[4,2]),
                    'CU_09F_Std': str(val_ciede[4,3]),
                    'CU_09F_Range': str(val_ciede[4,0] - val_ciede[4,1]),
                    'CU_DeltaE_Range': str(val_ciede[:, 0].max() - val_ciede[:, 1].min()),
    }
    if calcu_delta_c or calcu_old_cu:
        r_img, gr_img, gb_img, b_img = utils.split_channel(image, bayer_pattern)    
        image_height, image_width = image_size
        
        if calcu_delta_c:
            deltaCRange, deltaRG_GB = _delta_c(r_img, gr_img, gb_img, b_img, lab_full, lab_center, image_height, image_width, roi_size[0], mask_radius)
            data['CU_DeltaC_Range'] = deltaCRange
            data['CU_Delta_RG_BG_Range'] = deltaRG_GB

        if calcu_old_cu:
            mx, CU_Ratio_RGr, CU_Ratio_RGb, CU_Ratio_BGr, CU_Ratio_BGb = _old_cu(r_img, gr_img, gb_img, b_img)
            data['CU_Ratio_Max'] = mx
            data['CU_Ratio_RGr '] = CU_Ratio_RGr
            data['CU_Ratio_RGb'] = CU_Ratio_RGb
            data['CU_Ratio_BGr'] = CU_Ratio_BGr
            data['CU_Ratio_BGb '] = CU_Ratio_BGb
    
    if debug_flag or csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        
    if debug_flag:
        _debug_image(image, rgb, mask_radius, num_Ring, all_roi_rect, val_ciede, index_max_min, fov_rings, save_path)
            
    if csv_output:
        save_file_path = os.path.join(save_path, 'cu_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))
    return data

def func(cu_path, save_path, config_path):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    cu_cfg = cfg.color_uniformity
    images = utils.load_images(cu_path, cu_cfg.image_count, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)
    image = np.round(images.mean(axis=2)).astype(np.uint16)
    if cu_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    if cu_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)
    color_uniformity(image, cu_cfg.input_pattern, cu_cfg.roi_size, cu_cfg.mask_radius, cu_cfg.fov_rings, cu_cfg.calcu_delta_c, cu_cfg.calcu_old_cu, cu_cfg.csv_output, cu_cfg.debug_flag, save_path)
    return True

if __name__ == '__main__':
    file_name = r'C:\Users\wangjianan\Desktop\Innorev_Result\Lightfield\images'
    save_path = r'C:\Users\wangjianan\Desktop\Innorev_Result\Lightfield'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    utils.process_files(file_name, func, '.raw', save_path, config_path)
    print('CU finished!')












