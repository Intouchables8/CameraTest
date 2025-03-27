from Common import utils
from Project.ET import relative_illumination
from Project.ET import optical_center
from Project.ET import defect_pixel_light
from Project.ET import relative_uniformity
from Project.ET import defect_row_col_light
from Project.ET import tve_blemish

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

    # @time_it_avg(10)
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        
        if self.image_cfg.sub_black_level:
            sub_image = utils.sub_black_level(image, self.image_cfg.black_level)
        
        if self.image_cfg.bayer_pattern != 'Y':
            sub_y_image = utils.bayer_2_y(sub_image, self.image_cfg.bayer_pattern)
        else:
            sub_y_image = sub_image
        
        
        # relative illumination
        relative_illumination.relative_illumination(sub_y_image, self.ri_cfg.roi_size, self.ri_cfg.snr_roi_size, self.ri_cfg.mask_radius, self.ri_cfg.border_distance,
                            self.ri_cfg.csv_output, self.ri_cfg.debug_flag, save_path)
        
        ## OC的阈值表述不一致
        # optical center
        # center = sub_y_image[180: 200, 180: 200]
        # a = sub_y_image[: 40, : 40].mean()
        # b = sub_y_image[: 40, -40:].mean()
        # c = sub_y_image[-40:, : 40].mean()
        # d = sub_y_image[-40:, -40:].mean()
        # thresh = 0.5 * (center + np.array((a,b,c,d)).mean())
        optical_center.optical_center(sub_y_image, self.oc_cfg.thresh, self.oc_cfg.csv_output, save_path)
                
        # defect pixel light
        defect_pixel_light.defect_pixel_light(sub_y_image, self.dpl_cfg.input_pattern, self.dpl_cfg.roi_size, self.dpl_cfg.mask_radius, self.dpl_cfg.thresh,
                        self.dpl_cfg.csv_output, self.dpl_cfg.debug_flag, save_path)

        # DRCL   是否加掩膜
        defect_row_col_light.defect_row_col_light(sub_y_image, self.drcl_cfg.thresh, self.drcl_cfg.input_pattern, self.drcl_cfg.mask_radius, 
                             self.drcl_cfg.min_pixel, self.drcl_cfg.neighbor, save_path, self.drcl_cfg.csv_output, self.drcl_cfg.debug_flag)
        
        # relative_uniformity
        relative_uniformity.relative_uniformity(sub_y_image, self.ru_cfg.mask_radius, self.ru_cfg.roi_size, self.ru_cfg.delta_angle, self.ru_cfg.border_distance, self.ru_cfg.csv_output, self.ru_cfg.debug_flag, save_path)
    
        # blemish
        self.blemish.run(sub_y_image, save_path)

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\ET\light.raw'
    save_path = r'E:\Wrok\ERS\Diamond ET\Module Images (for algo correlation)\Dukono (non-POR)\Light\result'
    config_path = r'G:\CameraTest\Config\config_et.yaml'
    light = Light(config_path)
    # utils.process_file_or_folder(file_name, '.raw', light.func, save_path)
    
    light.func(file_name, save_path)

    print('light finished!') 
        
        
    