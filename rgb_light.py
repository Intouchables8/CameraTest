import sys
import os
ROOTPATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(ROOTPATH))
from Common import utils
from Project.RGB import relative_illumination
from Project.RGB import optical_center
from Project.RGB import defect_pixel_light
from Project.RGB import relative_uniformity
from Project.RGB import defect_row_col_light
from Project.RGB import tve_blemish
from Project.RGB import color_uniformity
import numpy as np
import os

class Light:
    def __init__(self, config_path):
        cfg = utils.load_config(config_path).light
        self.image_cfg = cfg.image_info
        self.ri_cfg = cfg.relative_illumination
        self.oc_cfg = cfg.optical_center
        self.drcl_cfg = cfg.defect_row_col_light
        self.dpl_cfg = cfg.defect_pixel_light
        self.ru_cfg = cfg.relative_uniformity
        self.blemish = tve_blemish.TVEBlemish(config_path)
        self.cu_cfg = cfg.color_uniformity


    # @time_it_avg(10)
    def func(self, file_name, save_path):
        images = utils.load_images(file_name, self.cu_cfg.image_count,self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        if self.image_cfg.sub_black_level:
            sub_images = utils.sub_black_level(images, self.image_cfg.black_level)
        
        sub_y_image = sub_images[:, :, 0].astype(np.uint16)
        if self.image_cfg.bayer_pattern != 'Y':
            sub_y_image = utils.bayer_2_y(sub_y_image, self.image_cfg.bayer_pattern)


        # relative illumination
        ri_data = relative_illumination.relative_illumination(sub_y_image, self.ri_cfg.roi_size, self.ri_cfg.snr_roi_size, self.ri_cfg.mask_radius, self.ri_cfg.border_distance,
                            self.ri_cfg.csv_output, self.ri_cfg.debug_flag, save_path)

        oc_data = optical_center.optical_center(sub_y_image, self.oc_cfg.thresh, self.oc_cfg.csv_output, save_path)
                
        # defect pixel light
        dpl_data = defect_pixel_light.defect_pixel_light(sub_y_image, self.dpl_cfg.input_pattern, self.dpl_cfg.roi_size, self.dpl_cfg.mask_radius, self.dpl_cfg.thresh,
                        self.dpl_cfg.csv_output, self.dpl_cfg.debug_flag, save_path)

        # DRCL   
        drcl_data = defect_row_col_light.defect_row_col_light(sub_y_image, self.drcl_cfg.thresh, self.drcl_cfg.input_pattern, self.drcl_cfg.mask_radius, 
                             self.drcl_cfg.min_pixel, self.drcl_cfg.neighbor, save_path, self.drcl_cfg.csv_output, self.drcl_cfg.debug_flag)
        
        # relative_uniformity
        ru_data = relative_uniformity.relative_uniformity(sub_y_image, self.ru_cfg.mask_radius, self.ru_cfg.roi_size, self.ru_cfg.delta_angle, self.ru_cfg.border_distance, self.ru_cfg.csv_output, self.ru_cfg.debug_flag, save_path)

        # blemish
        blemish_data = self.blemish.run(sub_y_image, save_path)
        
        # cu
        cu_data = color_uniformity.color_uniformity(sub_images.mean(axis=2), self.cu_cfg.input_pattern, self.cu_cfg.roi_size, self.cu_cfg.mask_radius, self.cu_cfg.fov_rings, 
                         self.cu_cfg.calcu_delta_c, self.cu_cfg.calcu_old_cu, self.cu_cfg.csv_output, self.cu_cfg.debug_flag, save_path)


# def find_first_raw_file(root_folder):
#     results = {}

#     # 遍历主文件夹下的所有子文件夹
#     # for subfolder in root_path.glob("**/light_files"):  # 查找所有名为 'sfr' 的文件夹
#     for subfolder in root_path.glob("*"):  # 查找所有名为 'sfr' 的文件夹
#         if subfolder.is_dir():  # 确保是目录
#             raw_files = sorted(subfolder.glob("*.raw"))  # 获取 .raw 文件，并排序
#             if raw_files:  # 确保存在 .raw 文件
#                 results[subfolder] = raw_files[0]  # 记录第一个 .raw 文件路径

#     return results

if __name__ == '__main__':
    file_name = r'D:\Code\CameraTest\image\RGB\light\20241221_144804__0_AS_DNPVerify_377TT04G9L01TG.raw'
    save_path = r'E:\Wrok\ERS\Diamond RGB\Module Images (for algo correlation)\Light (Fail)'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    light = Light(config_path)
    # utils.process_files(file_name, light.func, '.raw', cu_path, save_path)
    # raw_file_dict = find_first_raw_file(file_name)
    light.func(file_name, save_path)
    # for folder, file in raw_file_dict.items():
    #     light.func(file, save_path)

    print('light finished!') 
        
        
    