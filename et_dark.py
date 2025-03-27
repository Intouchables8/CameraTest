from Project.ET import dark_noise
from Project.ET import defect_pixel_dark
from Project.ET import DSNU
from Project.ET import defect_row_col_dark
from Common import utils
import numpy as np
class Dark:
    def __init__(self, config_path):
        cfg = utils.load_config(config_path).dark
        self.image_cfg = cfg.image_info
        self.dpd_cfg = cfg.defect_pixel_dark
        self.dn_cfg = cfg.dark_noise
        self.drcd_cfg = cfg.defect_row_col_dark
        self.dsnu_cfg = cfg.DSNU

    def func(self, dpd_file_name, dn_file_name=None, save_path='.result'):
        if dn_file_name is None or len(dn_file_name) == 0:
            dn_images = utils.load_images(dpd_file_name, self.dn_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
            if self.dpd_cfg.image_count > 1:
               dpd_image = np.round(dn_images.mean(axis=2)).astype(np.uint16) 
            else:
                dpd_image = dn_images[:, :, 0]
        else:
            dn_images = utils.load_images(dn_file_name, self.dn_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
            if self.dpd_cfg.image_count > 1:
                dpd_images = utils.load_images(dpd_file_name, self.dpd_cfg.image_count, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
                dpd_image = np.round(dpd_images.mean(axis=2)).astype(np.uint16)
            else:
                dpd_image = utils.load_images(dpd_file_name, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        # defect pixel dark
        defect_pixel_dark.defect_pixel_dark(dpd_image, self.dpd_cfg.input_pattern, self.dpd_cfg.thresh, self.dpd_cfg.csv_output, self.dpd_cfg.debug_flag, save_path)

        # dsnu
        DSNU.dsnu(dpd_image, self.dsnu_cfg.roi_size, self.dsnu_cfg.except_dpd, None, None, self.dsnu_cfg.csv_output, save_path)

        # defect row col dark
        defect_row_col_dark.defect_row_col_dark(dpd_image, self.drcd_cfg.thresh, self.dpd_cfg.input_pattern, self.drcd_cfg.neighbor, save_path, self.drcd_cfg.csv_output, self.drcd_cfg.debug_flag)
        
        # dark noise
        dark_noise.dark_noise(dn_images, self.dn_cfg.csv_output, save_path)    
        return 'dark finished'
            

if __name__ == '__main__':
    dpd_file_name = r'E:\Wrok\ERS\Diamond ET\Module Images (for algo correlation)\Dukono (non-POR)\Dark\10f10c010b59c27903023c0b4500400100000000_20241117_092340_Test_0.raw'
    dn_file_name = None
    save_path = r'E:\Wrok\ERS\Diamond ET\Module Images (for algo correlation)\Dukono (non-POR)\Dark\result'
    config_path = r'G:\CameraTest\Config\config_et.yaml'
    dark = Dark(config_path)
    # utils.process_file_or_folder(dpd_file_name, '.raw', dark.func, dn_file_name, save_path)
    dark.func(dpd_file_name, dn_file_name, save_path)
    print('dark finished!') 
        
        
    