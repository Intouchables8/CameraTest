from Common import utils
from Project.ET import sfr
from Customize import sfr_funcion
import numpy as np
import os
class CalSFR:
    def __init__(self, config_path):
        self.sfr = sfr.SFR(config_path)
        self.params = utils.load_config(config_path).sfr.customized_params
        
    
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.sfr.image_tpye, self.sfr.image_size, self.sfr.crop_tblr)
        if self.sfr.sub_black_level:
            image = utils.sub_black_level(image, self.sfr.black_level)
        
        if self.sfr.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.sfr.sfr.bayer_pattern)
        
        if image.dtype == np.uint16:
            image = (image >> 2).astype(np.uint8)
        
        # 定位block
        block_roi_center_xy, block_centroid, _, _ = self.sfr.localte_block_california(image, str(save_path))
        
        # 定位points
        chart_center_xy = block_centroid[0][0]
        points_xy = sfr_funcion.select_point_et(image, chart_center_xy, self.sfr.point_thresh, self.params.points_offset_xy, self.params.points_roi_size, self.sfr.n_point, self.sfr.point_dist_from_center, self.sfr.clockwise, self.sfr.debug_flag)
        
        # rotation
        rotation_tl_br, rotation_tr_bl, rotation_mean = sfr_funcion.rotation_et(points_xy)
        
        # oc
        center_xy = [image.shape[1] // 2, image.shape[0] // 2]
        offset_x, offset_y, oc_r = sfr_funcion.pointing_oc_et(points_xy, center_xy)
        
        # fov
        fov_d = sfr_funcion.fov_et(points_xy, self.params.image_circle, self.params.fov_design, self.params.fov_ratio)
        
        # 选择roi
        all_roi_center_xy = self.sfr.select_roi(block_roi_center_xy, self.sfr.roi_index)
        
        # 创建rect
        all_roi_rect = self.sfr.get_roi_rect(image, all_roi_center_xy)
            
        # 计算mtf
        self.sfr.calcu_mtf(image, all_roi_rect, str(save_path))
        
        degree_2_rad = 180 / np.pi
        data = {
                f'OC_Pointing_X': offset_x,
                f'OC_Pointing_Y': offset_y,
                f'OC_Pointing_R': oc_r,
                f'Rotation_TL_BR': rotation_tl_br * degree_2_rad,  
                f'Rotation_TR_BL': rotation_tr_bl * degree_2_rad,  
                f'Rotation_Mean': rotation_mean * degree_2_rad,  
                f'D_FOV': fov_d,  
                f'P5_x': points_xy[0][0],  
                f'P5_y': points_xy[0][1],  
                f'P6_x': points_xy[1][0],  
                f'P6_y': points_xy[1][1],  
                f'P17_x': points_xy[2][0],  
                f'P17_y': points_xy[2][1],  
                f'P18_x': points_xy[3][0],  
                f'P18_y': points_xy[3][1],  
        }
        device_id = utils.GlobalConfig.get_device_id()
        os.makedirs(save_path, exist_ok=True)
        save_file_name = os.path.join(save_path, (device_id + '.csv'))
        utils.save_dict_to_csv(data, str(save_file_name))
        
        
if __name__ == '__main__':
    file_name = r'.\image\ET\sfr.raw'
    save_path = r'.\result'
    config_path = r'G:\CameraTest\Config\config_et.yaml'
    sfr = CalSFR(config_path)
    sfr.func(file_name, save_path)
    # utils.process_file_or_folder(file_name, '.raw', sfr.func, save_path)
    # utils.process_file_or_folder(file_name, '.raw', sfr.func, save_path)
    print('sfr finished!') 
