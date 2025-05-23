import sys
import os
ROOTPATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(ROOTPATH))
from Common import utils
from Project.CV import relative_illumination
from Project.CV import optical_center
from Project.CV import defect_pixel_light
from Project.CV import relative_uniformity
from Project.CV import defect_row_col_light
# from Project.CV import tve_blemish
from Project.CV import dark_corner
from Project.CV import PRNU
import numpy as np

class Light:
    def __init__(self, config_path):
        cfg = utils.load_config(config_path).light
        self.image_cfg = cfg.image_info
        self.ri_cfg = cfg.relative_illumination
        self.oc_cfg = cfg.optical_center
        self.drcl_cfg = cfg.defect_row_col_light
        self.dpl_cfg = cfg.defect_pixel_light
        self.ru_cfg = cfg.relative_uniformity
        # self.blemish = tve_blemish.TVEBlemish(config_path)
        self.dc_cfg = cfg.dark_corner
        self.prnu_cfg = cfg.PRNU

    # @time_it_avg(10)
    def func(self, file_name, save_path):
        images = utils.load_images(file_name, self.prnu_cfg.image_count,self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        if self.image_cfg.sub_black_level:
            sub_images = utils.sub_black_level(images, self.image_cfg.black_level)
        
        sub_y_image = sub_images[:, :, 0].astype(np.uint16)

        # PRNU
        PRNU.prnu(sub_images, self.prnu_cfg.roi_size, self.prnu_cfg.csv_output, self.prnu_cfg.debug_flag, save_path)
        
        # relative illumination
        relative_illumination.relative_illumination(sub_y_image, self.ri_cfg.roi_size, self.ri_cfg.snr_roi_size, self.ri_cfg.mask_radius, self.ri_cfg.border_distance,
                            self.ri_cfg.csv_output, self.ri_cfg.debug_flag, save_path)

        # dark corner
        dark_corner.dark_corner(sub_y_image, self.dc_cfg.roi_size, self.dc_cfg.mask_radius, self.dc_cfg.border_distance,
                            self.dc_cfg.csv_output, self.dc_cfg.debug_flag, save_path)

        optical_center.optical_center(sub_y_image, self.oc_cfg.thresh, self.oc_cfg.csv_output, save_path)
                
        # defect pixel light
        defect_pixel_light.defect_pixel_light(sub_y_image, self.dpl_cfg.input_pattern, self.dpl_cfg.roi_size, self.dpl_cfg.mask_radius, self.dpl_cfg.thresh,
                        self.dpl_cfg.csv_output, self.dpl_cfg.debug_flag, save_path)

        # DRCL   是否加掩膜
        defect_row_col_light.defect_row_col_light(sub_y_image, self.drcl_cfg.thresh, self.drcl_cfg.input_pattern, self.drcl_cfg.mask_radius, 
                             self.drcl_cfg.min_pixel, self.drcl_cfg.neighbor, save_path, self.drcl_cfg.csv_output, self.drcl_cfg.debug_flag)
        
        # relative_uniformity
        relative_uniformity.relative_uniformity(sub_y_image, self.ru_cfg.mask_radius, self.ru_cfg.roi_size, self.ru_cfg.delta_angle, self.ru_cfg.border_distance, self.ru_cfg.csv_output, self.ru_cfg.debug_flag, save_path)

        # # blemish
        # self.blemish.run(sub_y_image, save_path)


# def find_first_raw_file(root_folder):
#     root_path = Path(root_folder)
#     results = {}

#     # 遍历主文件夹下的所有子文件夹
#     for subfolder in root_path.glob("**/light_files"):  # 查找所有名为 'sfr' 的文件夹
#         if subfolder.is_dir():  # 确保是目录
#             raw_files = sorted(subfolder.glob("*.raw"))  # 获取 .raw 文件，并排序
#             if raw_files:  # 确保存在 .raw 文件
#                 results[subfolder] = raw_files[0]  # 记录第一个 .raw 文件路径
#     return results
if __name__ == '__main__':
    file_name = r'D:\Code\CameraTest\image\CV\light\Ketron_P0C_FF2_Line1_Light1_EOL-Light__030703111601010e0b0300001a08_20241229041233_0.raw'
    save_path = r'D:\Code\CameraTest\result'
    config_path = r'D:\Code\CameraTest\Config\config_cv.yaml'
    light = Light(config_path)
    light.func(file_name, save_path)
    # utils.process_files(file_name, '.raw', light.func, cu_path, save_path)
    # raw_file_dict = find_first_raw_file(file_name)
    # for folder, file in raw_file_dict.items():
    #     light.func(file, save_path)

    print('light finished!') 
        
        
    