from Common import utils
from Script.relative_illumination import relative_illumination
from Script.optical_center import optical_center
from Script.defect_pixel_light import defect_pixel_light
from Script.relative_uniformity import relative_uniformity
from Script.defect_row_col_light import defect_row_col_light
from Script.tve_blemish import TVEBlemish
from Script.color_uniformity import color_uniformity
from Common.utils import *

class Light:
    def __init__(self, config_path):
        config_path = Path(config_path)
        cfg = utils.load_config(config_path).light
        self.image_cfg = cfg.image_info
        self.ri_cfg = cfg.relative_illumination
        self.oc_cfg = cfg.optical_center
        self.drcl_cfg = cfg.defect_row_col_light
        self.dpl_cfg = cfg.defect_pixel_light
        self.ru_cfg = cfg.relative_uniformity
        self.blemish = TVEBlemish(config_path)
        self.cu_cfg = cfg.color_uniformity


    # @time_it_avg(10)
    def func(self, file_name, save_path):
        save_path = Path(save_path)
        file_name = Path(file_name)
        images = utils.load_images(file_name, self.cu_cfg.image_count,self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        if self.image_cfg.sub_black_level:
            sub_images = utils.sub_black_level(images, self.image_cfg.black_level)
        
        sub_y_image = sub_images[:, :, 0].astype(np.uint16)
        if self.image_cfg.bayer_pattern != 'Y':
            sub_y_image = utils.bayer_2_y(sub_y_image, self.image_cfg.bayer_pattern)


        # relative illumination
        relative_illumination(sub_y_image, self.ri_cfg.roi_size, self.ri_cfg.snr_roi_size, self.ri_cfg.mask_radius, self.ri_cfg.border_distance,
                            self.ri_cfg.csv_output, self.ri_cfg.debug_flag, save_path)

        optical_center(sub_y_image, self.oc_cfg.thresh, self.oc_cfg.csv_output, save_path)
                
        # defect pixel light
        defect_pixel_light(sub_y_image, self.dpl_cfg.input_pattern, self.dpl_cfg.roi_size, self.dpl_cfg.mask_radius, self.dpl_cfg.thresh,
                        self.dpl_cfg.csv_output, self.dpl_cfg.debug_flag, save_path)

        # DRCL   
        defect_row_col_light(sub_y_image, self.drcl_cfg.thresh, self.drcl_cfg.input_pattern, self.drcl_cfg.mask_radius, 
                             self.drcl_cfg.min_pixel, self.drcl_cfg.neighbor, save_path, self.drcl_cfg.csv_output, self.drcl_cfg.debug_flag)
        
        # relative_uniformity
        relative_uniformity(sub_y_image, self.ru_cfg.mask_radius, self.ru_cfg.roi_size, self.ru_cfg.delta_angle, self.ru_cfg.border_distance, self.ru_cfg.csv_output, self.ru_cfg.debug_flag, save_path)

        # blemish
        self.blemish.run(sub_y_image, save_path)
        
        # cu
        color_uniformity(sub_images.mean(axis=2), self.cu_cfg.input_pattern, self.cu_cfg.roi_size, self.cu_cfg.mask_radius, self.cu_cfg.fov_rings, 
                         self.cu_cfg.calcu_delta_c, self.cu_cfg.calcu_old_cu, self.cu_cfg.csv_output, self.cu_cfg.debug_flag, save_path)


def find_first_raw_file(root_folder):
    root_path = Path(root_folder)
    results = {}

    # 遍历主文件夹下的所有子文件夹
    # for subfolder in root_path.glob("**/light_files"):  # 查找所有名为 'sfr' 的文件夹
    for subfolder in root_path.glob("*"):  # 查找所有名为 'sfr' 的文件夹
        if subfolder.is_dir():  # 确保是目录
            raw_files = sorted(subfolder.glob("*.raw"))  # 获取 .raw 文件，并排序
            if raw_files:  # 确保存在 .raw 文件
                results[subfolder] = raw_files[0]  # 记录第一个 .raw 文件路径

    return results

if __name__ == '__main__':
    file_name = r'E:\Wrok\ERS\Diamond RGB\Module Images (for algo correlation)\Light (Fail)\class'
    save_path = r'E:\Wrok\ERS\Diamond RGB\Module Images (for algo correlation)\Light (Fail)'
    config_path = r'G:\CameraTest\Config\config_rgb.yaml'
    light = Light(config_path)
    # utils.process_file_or_folder(file_name, '.raw', light.func, cu_path, save_path)
    raw_file_dict = find_first_raw_file(file_name)
    for folder, file in raw_file_dict.items():
        light.func(file, save_path)

    print('light finished!') 
        
        
    