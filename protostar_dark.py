import sys
import os
ROOTPATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(ROOTPATH))
from Project.RGB import dark_noise
from Project.RGB import defect_pixel_dark
from Project.RGB import DSNU
from Project.RGB import defect_row_col_dark
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

    def func(self, dpd_file_name, dn_file_name=None, save_path='/result'):
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
        dpd_data, dpd_map = defect_pixel_dark.defect_pixel_dark(dpd_image, self.dpd_cfg.input_pattern, self.dpd_cfg.thresh, self.dpd_cfg.csv_output, self.dpd_cfg.debug_flag, save_path)
        
        # DSNU
        dsnu_data = DSNU.dsnu(dn_images.mean(axis=2), self.dsnu_cfg.roi_size, self.dsnu_cfg.except_dpd, dpd_map, self.dpd_cfg.thresh, self.dsnu_cfg.csv_output, save_path)
        
        # defect row col dark
        drcd_data = defect_row_col_dark.defect_row_col_dark(dpd_image, self.drcd_cfg.thresh, self.dpd_cfg.input_pattern, self.drcd_cfg.neighbor, save_path, self.drcd_cfg.csv_output, self.drcd_cfg.debug_flag)
        
        # dark noise
        dn_data = dark_noise.dark_noise(dn_images, self.dn_cfg.csv_output, save_path)    
        
            

# def find_first_raw_file(root_folder):
#     root_path = Path(root_folder)
#     results = {}

#     # 遍历主文件夹下的所有子文件夹
#     for subfolder in root_path.glob("**/dark_files"):  # 查找所有名为 'sfr' 的文件夹
#         if subfolder.is_dir():  # 确保是目录
#             raw_files = sorted(subfolder.glob("*.raw"))  # 获取 .raw 文件，并排序
#             if raw_files:  # 确保存在 .raw 文件
#                 results[subfolder] = raw_files[0]  # 记录第一个 .raw 文件路径

#     return results

if __name__ == '__main__':
    dpd_file_name = r'D:\Code\CameraTest\image\CV\dark\Ketron_P0C_FF2_Line1_DARK1_EOL-Dark_373KQ11GC300V8_030703111601010e0b0300001a08_20241228153651_0.raw'
    dn_file_name = None
    save_path = r'D:\Code\CameraTest\result'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    dark = Dark(config_path)
    dark.func(dpd_file_name, dn_file_name, save_path)
    # # utils.process_files(dpd_file_name, dark.func, '.raw', dn_file_name, save_path)
    # raw_file_dict = find_first_raw_file(dpd_file_name)
    # for folder, file in raw_file_dict.items():
    #     dark.func(file, dn_file_name, save_path)

    print('dark finished!') 
        
        
    