from Script.dark_noise import dark_noise
from Script.defect_pixel_dark import defect_pixel_dark
from Common import utils
import numpy as np
from Common.utils import *
class Dark:
    def __init__(self, config_path):
        config_path = Path(config_path)
        cfg = utils.load_config(config_path).dark
        self.image_cfg = cfg.image_info
        self.dpd_cfg = cfg.defect_pixel_dark
        self.dn_cfg = cfg.dark_noise

    @time_it_avg(10)
    def func(self, dpd_file_name, dn_file_name=None, save_path=None):
        dpd_file_name = Path(dpd_file_name)
        if dn_file_name is None:
            images = utils.load_images(dpd_file_name, self.dn_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
            if self.dpd_cfg.image_count > 1:
                image = images.mean(axis=2)
            else:
                image = images[:,:,0]
        else:
            dn_file_name = Path(dn_file_name)
            images = utils.load_images(dn_file_name, self.dn_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)

            if self.dpd_cfg.image_count > 1:
                images = utils.load_images(dpd_file_name, self.dpd_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
                image = np.round(images.mean(axis=2)).astype(np.uint16)
            else:
                image = utils.load_images(dpd_file_name, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)

            defect_pixel_dark(image, self.dpd_cfg.input_pattern, self.dpd_cfg.roi_size, self.dpd_cfg.thresh, self.dpd_cfg.csv_output, self.dpd_cfg.debug_flag, save_path)
            
            dark_noise(images, self.dn_cfg.csv_output, save_path)    
            return 'dark finished'
            

if __name__ == '__main__':
    dpd_file_name = r'G:\CameraTest\image\california\Dark\dpd\California_P0_DARK_1_2_Dark16X_352RK1AFBV00K5_3660681a28230823610100_20231226_122724_0.raw'
    dn_file_name = r'G:\CameraTest\image\california\Dark\noise\California_P0_DARK_1_2_Dark16X_352RK1AFBV00K5_3660681a28230823610100_20231226_122724_0.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_california.yaml'
    dark = Dark(config_path)
    dark.func(dpd_file_name, dn_file_name, save_path)
    print('dark finished!') 
        
        
    