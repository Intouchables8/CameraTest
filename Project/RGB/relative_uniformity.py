import numpy as np
import os
import cv2
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(str(ROOTPATH))
from Common import utils

def _debug_image(image, all_rect, all_value, local_delta, vec_ru_slop, ring_radius, local_avg, save_path):
    rgb = utils.raw_2_rgb(image)
    image_size = image.shape
    if False:
        # 生成热力图_ --------------------------------------------------
        heat_map = (255 * (image / image.max())).astype('uint8')
        for ring in all_rect:
            for pos in ring:
                cv2.rectangle(heat_map, (pos[0], pos[1]), (pos[2], pos[3]), 0)
        import matplotlib.pyplot as plt
        plt.ioff() # 关闭交互模式 这样不显示
        plt.imshow(heat_map, cmap='jet')
        heat_map_save_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_RU_HeatMap.png'))
        
        plt.savefig(heat_map_save_path, dpi=450)
        plt.close
        #------------------------------------------------------------
    maxRing = local_delta.argmax()
    val_ring_data = np.array(all_value[maxRing])
    max_data = val_ring_data.max()
    min_data = val_ring_data.min()
    
    # 标记RU SLOP MAX-------------------------------------------------
    max_slop_max = vec_ru_slop.argmax()
    maskRadius_slop = ring_radius[max_slop_max: max_slop_max+2].mean()
    cv2.circle(rgb, (image_size[1] // 2, image_size[0] // 2), int(maskRadius_slop), color=(255, 255, 0))
    #-----------------------------------------------
    cur_ring = 0
    for ring in all_rect:
        box_max = np.array(all_value[cur_ring]).argmax()
        box_min = np.array(all_value[cur_ring]).argmin()
        max_data = np.round(np.array(all_value[cur_ring]).max(), 2)
        min_data = np.round(np.array(all_value[cur_ring]).min(), 2)
        
        if cur_ring == maxRing: # 标记local delta 最大的一环
            
            cur_box = 0
            j = 1
            for pos in ring:
                cv2.putText(rgb, str(j), (pos[0]+2, pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (182, 110, 0), 1)
                j = j + 1
                if cur_box == box_max:
                    text = 'Max: ' + str(max_data)
                    cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 255))
                    cv2.putText(rgb, text, (pos[0] - 10, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cur_box = cur_box + 1
                else:
                    if cur_box == box_min:    
                        text = 'Min: ' + str(min_data)
                        cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0))
                        cv2.putText(rgb, text, (pos[0] - 10, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cur_box = cur_box + 1
                    else:    
                        cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 255))
                        cur_box = cur_box + 1
            cur_ring = cur_ring + 1
            
        else:
            cur_box = 0
            
            i = 1 # ROI 数量标记
            for pos in ring:
                if cur_box == box_max:
                    text = 'Max: ' + str(max_data)
                    cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 255))
                    cv2.putText(rgb, text, (pos[0] - 10, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cur_box = cur_box + 1
                else:
                    if cur_box == box_min:    
                        text = 'Min: ' + str(min_data)
                        cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0))
                        cv2.putText(rgb, text, (pos[0] - 10, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cur_box = cur_box + 1
                    else:    
                        cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (150, 0, 0))
                        
                        cur_box = cur_box + 1
                # cv2.rectangle(rgb, (pos[0], pos[1]), (pos[2], pos[3]), (150, 0, 0))
                # cv2.putText(rgb, str(i), (pos[0]+2, pos[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (182, 110, 0), 1)
                # i = i + 1
            cur_ring = cur_ring + 1
    for i in range(len(local_delta)):
        x = 10
        y = 10 + 15*i
        # 增加文字框背景
        background = np.zeros_like(rgb)
        background[y - 9: y+2, x - 3: x+180] = (0, 255, 255)
        rgb = cv2.addWeighted(rgb, 1, background, 0.5, 0)
        
        text = 'cir' + str(i+1) + ' Delta:' + str(round(local_delta[i],3)) + ' Avg:' + str(round(local_avg[i],3))
        cv2.putText(rgb, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    save_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_RU.png'))
    cv2.imwrite(save_path, rgb)

def _get_ring_roi_value(y_image, roi_center, roi_height):
    half_roi_height = round(roi_height/2)
    n_ring_box = len(roi_center)
    ring_rect = np.zeros((n_ring_box, 4), np.uint16)
    ring_value = np.zeros((n_ring_box), np.float32)
    for j in range(n_ring_box):
        x_strat = roi_center[j][0] - half_roi_height
        x_end = x_strat + roi_height 
        y_start = roi_center[j][1] - half_roi_height
        y_end = y_start + roi_height 
        roi = y_image[y_start: y_end, x_strat: x_end]
        ring_value[j] = roi.mean()        
        ring_rect[j, :] = [x_strat, y_start, x_end, y_end]
    return ring_value, ring_rect

def relative_uniformity(src_image, mask_radius, roi_size, delta_angle, border_distance, csv_output, debug_flag=False, save_path=None):
    image = src_image.copy()
    image_size = image.shape
    mask = utils.generate_mask(image_size, mask_radius)
    image[mask == 0] = 0 
    roi_height = roi_size[0]
    num_ring = np.uint8(mask_radius / (2 * roi_height) -1)
    ring = np.arange(1, 1 + num_ring) # 第一圈到第n圈
    ring_radius = (2 * ring + 1) * roi_height
    angle = delta_angle / ring # 每一圈的角度差
    # angle = np.clip(angle, 1, None) # 限制角度差，如果角度差小于一度，就选用一度 
    center_xy = [image_size[1] // 2, image_size[0] // 2]
    all_rect = [None] * num_ring
    all_value = [None] * num_ring
    
    local_avg = np.zeros((num_ring), np.float32)
    local_delta = np.zeros((num_ring), np.float32)

    for i in range(int(num_ring)):
        cur_angle = angle[i]
        cur_radius = ring_radius[i]
        # 生成中心点坐标
        ring_roi_center =  utils.generate_circle(image_size, center_xy, cur_radius, cur_angle, roi_size, border_distance)
        ring_value, ring_rect = _get_ring_roi_value(image, ring_roi_center, roi_height)
        all_rect[i] = ring_rect
        all_value[i] = ring_value
        local_avg[i] = ring_value.mean()
        local_delta[i] = ring_value.max() - ring_value.min()

    local_ratio_delta = local_delta / local_avg
    # local delta
    ru_local_delta_max = local_delta.max()
    ru_local_delta_min = local_delta.min()
    ru_local_delta_avg = local_delta.mean()
    ru_local_delta_std = local_delta.std(ddof=1) # 这里使用样本标准差
    # ru slop
    vec_ru_slop = abs(local_avg[1:] - local_avg[:-1])
    ru_slop_max = vec_ru_slop.max()
    ru_slop_min = vec_ru_slop.min()
    ru_slop_avg = vec_ru_slop.mean()
    ru_slop_std = vec_ru_slop.std(ddof=1)
    
    # local ratio delta
    ru_local_ratio_delta_max = local_ratio_delta.max()
    ru_local_ratio_delta_min = local_ratio_delta.min()
    ru_local_ratio_delta_avg = local_ratio_delta.mean()
    ru_local_ratio_delta_std = local_ratio_delta.std(ddof=1) # 这里使用样本标准差
    # ru ratio slop
    vec_ru_ratio_slop = np.zeros_like(vec_ru_slop)
    for i in range(1, len(local_avg)):
        vec_ru_ratio_slop[i -1] = abs(local_avg[i] - local_avg[i-1]) / local_avg[i]

    ru_slop_ratio_max = vec_ru_ratio_slop.max()
    ru_slop_ratio_min = vec_ru_ratio_slop.min()
    ru_slop_ratio_avg = vec_ru_ratio_slop.mean()
    ru_slop_ratio_std = vec_ru_ratio_slop.std(ddof=1)
    
    data = {
                'RU_Local_Delta_Max': str(ru_local_delta_max),
                'RU_Local_Delta_Min': str(ru_local_delta_min),
                'RU_Local_Delta_Avg': str(ru_local_delta_avg),
                'RU_Local_Delta_Std': str(ru_local_delta_std),
                'RU_Slope_Max': str(ru_slop_max),
                'RU_Slope_Min': str(ru_slop_min),
                'RU_Slope_Avg': str(ru_slop_avg),
                'RU_Slope_Std': str(ru_slop_std),
                'RU_Local_Delta_Ratio_Max': str(ru_local_ratio_delta_max),
                'RU_Local_Delta_Ratio_Min': str(ru_local_ratio_delta_min),
                'RU_Local_Delta_Ratio_Avg': str(ru_local_ratio_delta_avg),
                'RU_Local_Delta_Ratio_Std': str(ru_local_ratio_delta_std),
                'RU_Slope_Ratio_Max': str(ru_slop_ratio_max),
                'RU_Slope_Ratio_Min': str(ru_slop_ratio_min),
                'RU_Slope_Ratio_Avg': str(ru_slop_ratio_avg),
                'RU_Slope_Ratio_Std': str(ru_slop_ratio_std)
    }
    if debug_flag or csv_output:
        
        os.makedirs(save_path, exist_ok=True)
    
    if debug_flag:
        _debug_image(image, all_rect, all_value, local_delta, vec_ru_slop, ring_radius, local_avg, save_path)
    
    if csv_output:
        save_file_path = os.path.join(save_path, 'ru_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))
    return data

def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    ru_cfg = cfg.relative_uniformity
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if ru_cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
    
    if ru_cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)

    relative_uniformity(image, ru_cfg.mask_radius, ru_cfg.roi_size, ru_cfg.delta_angle, ru_cfg.border_distance, ru_cfg.csv_output, ru_cfg.debug_flag, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\RGB\light.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_rgb.yaml'
    # import time 
    # for _ in range(10):
    #     start = time.time()
    #     func(file_name, save_path, config_path)
    #     print((time.time() - start))
    func(file_name, save_path, config_path)
    
    print('RU finished!')
    
