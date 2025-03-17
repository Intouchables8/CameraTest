from Common import utils
from Script.relative_illumination import relative_illumination
from Script.optical_center import optical_center
from Script.defect_pixel_light import defect_pixel_light
from Script.relative_uniformity import relative_uniformity
from Script.color_uniformity import color_uniformity
import numpy as np
from Common.utils import *

class Light:
    def __init__(self, config_path):
        config_path = Path(config_path)
        cfg = utils.load_config(config_path).light
        self.image_cfg = cfg.image_info
        self.ri_cfg = cfg.relative_illumination
        self.oc_cfg = cfg.optical_center
        self.dpl_cfg = cfg.defect_pixel_light
        self.ru_cfg = cfg.relative_uniformity
        self.cu_cfg = cfg.color_uniformity

    @time_it_avg(10)
    def func(self, file_name, cu_path, save_path):
        save_path = Path(save_path)
        file_name = Path(file_name)
        image = utils.load_image(file_name, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        if self.image_cfg.sub_black_level:
            sub_image = utils.sub_black_level(image, self.image_cfg.black_level)
        
        if self.image_cfg.bayer_pattern != 'Y':
            sub_y_image = utils.bayer_2_y(sub_image, self.image_cfg.bayer_pattern)
        else:
            sub_y_image = sub_image
        
        cur_image = None
        
        # relative illumination
        cur_image = sub_y_image
        relative_illumination(cur_image, self.ri_cfg.roi_size, self.ri_cfg.snr_roi_size, self.ri_cfg.mask_radius, self.ri_cfg.border_distance,
                            self.ri_cfg.csv_output, self.ri_cfg.debug_flag, save_path)
    
        # optical center
        cur_image = sub_y_image
        optical_center(cur_image, self.oc_cfg.thresh, self.oc_cfg.csv_output, save_path)
                
        # defect pixel light
        cur_image = sub_y_image
        defect_pixel_light(cur_image, self.dpl_cfg.input_pattern, self.dpl_cfg.roi_size, self.dpl_cfg.mask_radius, self.dpl_cfg.thresh,
                        self.dpl_cfg.csv_output, self.dpl_cfg.debug_flag, save_path)
      
        # relative_uniformity
        cur_image = sub_y_image
        relative_uniformity(cur_image, self.ru_cfg.mask_radius, self.ru_cfg.roi_size, self.ru_cfg.delta_angle, self.ru_cfg.border_distance, self.ru_cfg.csv_output, self.ru_cfg.debug_flag, save_path)
    
        # color uniformity
        images = utils.load_images(cu_path, self.cu_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        image = np.round(images.mean(axis=2)).astype(np.uint16)
        if self.cu_cfg.sub_black_level:
            image = utils.sub_black_level(image, self.image_cfg.black_level)
        
        if self.cu_cfg.input_pattern == 'Y' and self.image_cfg.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.image_cfg.bayer_pattern)
        color_uniformity(image, self.cu_cfg.input_pattern, self.cu_cfg.roi_size, self.cu_cfg.mask_radius, self.cu_cfg.fov_rings, self.cu_cfg.csv_output, self.cu_cfg.debug_flag, save_path)
        

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\california\Light.raw'
    cu_path = r'G:\CameraTest\image\california\CU\California_P0_DARK_1_2_Light_352RK1AFBV004K_3660681a28230914610100_20231226_161547_0.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_california.yaml'
    light = Light(config_path)
    light.func(file_name, cu_path, save_path)

    print('light finished!') 
        
        
    