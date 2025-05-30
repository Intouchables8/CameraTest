import sys
import os
ROOTPATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(ROOTPATH))
from Common import utils
from Project.RGB import sfr
import os
import numpy as np
from Customize import sfr_funcion

class CalSFR:
    def __init__(self, config_path):
        self.sfr = sfr.SFR(config_path, 3)
        self.cfg = self.sfr.cfg.customized_params
        
    
    def func(self, file_name, save_path, distance):
        image = utils.load_image(file_name, self.sfr.image_tpye, self.sfr.image_size, self.sfr.crop_tblr)
        if self.sfr.sub_black_level:
            image = utils.sub_black_level(image, self.sfr.black_level)
        
        if self.sfr.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.sfr.bayer_pattern)
        
        if image.dtype == np.uint16:
            image = (image >> 2).astype(np.uint8)
        
        image_size = image.shape
        # 定位block
        block_roi_center_xy, block_centroid, inner_block_center_xy, _ = self.sfr.localte_block_rgb(image, save_path)
        
        # 选择roi
        all_roi_center_xy = self.sfr.select_roi(block_roi_center_xy, self.sfr.roi_index)
        
        # 创建rect
        all_roi_rect = self.sfr.get_roi_rect(image, all_roi_center_xy)
            
        # 计算mtf
        mtf_data = self.sfr.calcu_mtf(image, all_roi_rect, save_path)
        
        # rotation
        point_xy_0_rotation = inner_block_center_xy[58]
        point_xy_1_rotation = inner_block_center_xy[59]
        point_xy_2_rotation = inner_block_center_xy[45]
        point_xy_3_rotation = inner_block_center_xy[57]
        point_xy_4_rotation = inner_block_center_xy[71]
        rotation, _, _, _, _ = sfr_funcion.rotation_rgb(point_xy_0_rotation, point_xy_1_rotation, point_xy_2_rotation, 
                                                        point_xy_3_rotation, point_xy_4_rotation)
        
        # fov
        point_xy_1_tilt = inner_block_center_xy[0]
        point_xy_2_tilt = inner_block_center_xy[12]
        point_xy_3_tilt = inner_block_center_xy[104]
        point_xy_4_tilt = inner_block_center_xy[116]
        fOv_h, fov_v, fov_d = sfr_funcion.fov_rgb(point_xy_1_tilt, point_xy_2_tilt, point_xy_3_tilt, point_xy_4_tilt, 
                                                  image_size, self.cfg.length_h_real_mm, self.cfg.length_v_real_mm, distance)
        
        # tilt
        tilt_x, tilt_y = sfr_funcion.tilt_rgb(point_xy_1_tilt, point_xy_2_tilt, point_xy_3_tilt, point_xy_4_tilt)
        
        # sfr AVG
        SFR_00F_AVG = mtf_data[:4].mean()
        SFR_20F_AVG = mtf_data[4: 12].mean()
        SFR_40F_AVG = mtf_data[12: 24].mean()
        SFR_60F_AVG = mtf_data[24: 44].mean()
        SFR_80F_AVG = mtf_data[44:].mean()
        
        # sfr Delta
        SFR_20F = np.array(mtf_data[4:6] + mtf_data[6:8] + mtf_data[8:10] +mtf_data[10:12])
        SFR_40F = np.array(mtf_data[12:14] + mtf_data[14:16] + mtf_data[16:18] +mtf_data[18:20])
        SFR_60F = np.array(mtf_data[24:26] + mtf_data[26:28] + mtf_data[28:30] +mtf_data[30:32] + mtf_data[32:34] + mtf_data[34:36] + mtf_data[36:38] +mtf_data[38:40])
        SFR_80F = np.array(mtf_data[44:46] + mtf_data[46:48] + mtf_data[48:50] +mtf_data[50:])
        SFR_20F_DELTA = SFR_20F.max() - SFR_20F.min()
        SFR_40F_DELTA = SFR_40F.max() - SFR_40F.min()
        SFR_60F_DELTA = SFR_60F.max() - SFR_60F.min()
        SFR_80F_DELTA = SFR_80F.max() - SFR_80F.min()
        
        data = {
                f'SFR_{distance}cm_00F_Avg_A_Ny4': SFR_00F_AVG, 
                f'SFR_{distance}cm_20F_Avg_A_Ny4': SFR_20F_AVG, 
                f'SFR_{distance}cm_40F_Avg_A_Ny4': SFR_40F_AVG, 
                f'SFR_{distance}cm_60F_Avg_A_Ny4': SFR_60F_AVG, 
                f'SFR_{distance}cm_80F_Avg_A_Ny4': SFR_80F_AVG, 
                f'SFR_{distance}cm_20F_Delta_A_Ny4': SFR_20F_DELTA, 
                f'SFR_{distance}cm_40F_Delta_A_Ny4': SFR_40F_DELTA, 
                f'SFR_{distance}cm_60F_Delta_A_Ny4': SFR_60F_DELTA, 
                f'SFR_{distance}cm_80F_Delta_A_Ny4': SFR_80F_DELTA, 
                
                f'SFR_{distance}cm_Rotation_Point_0_x': point_xy_0_rotation[0], 
                f'SFR_{distance}cm_Rotation_Point_0_y': point_xy_0_rotation[1], 
                f'SFR_{distance}cm_Rotation_Point_1_x': point_xy_1_rotation[0], 
                f'SFR_{distance}cm_Rotation_Point_1_y': point_xy_1_rotation[1], 
                f'SFR_{distance}cm_Rotation_Point_2_x': point_xy_2_rotation[0], 
                f'SFR_{distance}cm_Rotation_Point_2_y': point_xy_2_rotation[1], 
                f'SFR_{distance}cm_Rotation_Point_3_x': point_xy_3_rotation[0], 
                f'SFR_{distance}cm_Rotation_Point_3_y': point_xy_3_rotation[1], 
                f'SFR_{distance}cm_Rotation_Point_4_x': point_xy_4_rotation[0], 
                f'SFR_{distance}cm_Rotation_Point_4_y': point_xy_4_rotation[1], 
                f'SFR_{distance}cm_Rotation': np.degrees(rotation) ,
                
                f'SFR_{distance}cm_Tilt_Point_1_x': point_xy_1_tilt[0],
                f'SFR_{distance}cm_Tilt_Point_1_y': point_xy_1_tilt[1],
                f'SFR_{distance}cm_Tilt_Point_2_x': point_xy_2_tilt[0],
                f'SFR_{distance}cm_Tilt_Point_2_y': point_xy_2_tilt[1],
                f'SFR_{distance}cm_Tilt_Point_3_x': point_xy_3_tilt[0],
                f'SFR_{distance}cm_Tilt_Point_3_y': point_xy_3_tilt[1],
                f'SFR_{distance}cm_Tilt_Point_4_x': point_xy_4_tilt[0],
                f'SFR_{distance}cm_Tilt_Point_4_y': point_xy_4_tilt[1],
                f'SFR_{distance}cm_Tilt X': np.degrees(tilt_x),
                f'SFR_{distance}cm_Tilt Y': np.degrees(tilt_y),  
        }
        
        device_id = utils.GlobalConfig.get_device_id()
        os.makedirs(save_path, exist_ok=True)
        save_file_name = os.path.join(save_path, 'sfr.csv')
        utils.save_dict_to_csv(data, str(save_file_name))

            
if __name__ == '__main__':
    file_name = r'C:\Users\wangjianan\Desktop\Innorev_Result\SFR50'
    save_path = r'C:\Users\wangjianan\Desktop\Innorev_Result\SFR50'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    sfr = CalSFR(config_path)
    # 035 100 050 300
    distance = '050'
    utils.process_files(file_name, sfr.func, '.raw', save_path, distance)
    # sfr.func(file_name, save_path, distance)
    print('sfr finished!') 
