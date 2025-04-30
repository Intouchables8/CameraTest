from Common import utils
from Project.CV import sfr
from Customize import sfr_funcion
import numpy as np
import os
class CalSFR:
    def __init__(self, config_path):
        self.sfr = sfr.SFR(config_path, 1)
        self.cfg = self.sfr.cfg.customized_params
    
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.sfr.image_tpye, self.sfr.image_size, self.sfr.crop_tblr)
        if self.sfr.sub_black_level:
            image = utils.sub_black_level(image, self.sfr.black_level)
        
        if self.sfr.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.sfr.sfr.bayer_pattern)
        
        if image.dtype == np.uint16:
            image_8bit = (image >> 2).astype(np.uint8)
        
        # 定位block
        block_roi_center_xy, _, inner_blcok_centroid, bw_image = self.sfr.locate_block_cv(image_8bit, save_path)
        
        # 定位所有point相关坐标
        center_xy = (self.sfr.image_size[1] // 2, self.sfr.image_size[0] // 2)
        _, centroid =self.sfr.find_connected_area(bw_image, self.sfr.point_thresh, 'Point')
        index = np.argmin(utils.calcu_distance(centroid, center_xy))
        chart_center_xy = centroid[index]
        points_xy = sfr_funcion.select_point_cv(centroid, chart_center_xy, self.sfr.n_point, self.sfr.point_dist_from_center, self.sfr.clockwise, self.sfr.debug_flag)
    
        # 选择roi
        all_roi_center_xy = self.sfr.select_roi(block_roi_center_xy, self.sfr.roi_index)
        
        # 创建rect
        all_roi_rect = self.sfr.get_roi_rect(image_8bit, all_roi_center_xy)
            
        # 计算mtf
        mtf_data = self.sfr.calcu_mtf(image_8bit, all_roi_rect, save_path)
        sfr00f_min = mtf_data[0: 4, 0].min() * 100
        sfr03f_min = mtf_data[4: 12, 0].min() * 100
        sfr06f_min = mtf_data[12: 20, 0].min() * 100
        sfr08f_min = mtf_data[20: 28, 0].min() * 100
        sfr06f_balance = ((mtf_data[12: 20, 0].max() - mtf_data[12: 20, 0].min()) / mtf_data[12: 20, 0].max()) * 100
        sfr08f_balance = ((mtf_data[20: 28, 0].max() - mtf_data[20: 28, 0].min()) / mtf_data[20: 28, 0].max()) * 100
        
        # point
        #  b     a
        #     e 
        #  c     d
        point_a = inner_blcok_centroid[5]
        point_b = inner_blcok_centroid[4]
        point_c = inner_blcok_centroid[7]
        point_d = inner_blcok_centroid[6]
        #    g
        # h  e  k
        #    j
        point_e = points_xy[0]
        point_h = points_xy[1]
        point_g = points_xy[2]
        point_k = points_xy[3]
        point_j = points_xy[4]
        # OC
        points_xy = np.array([point_a, point_b, point_c, point_d, point_e])
        oc_x, oc_y, offset_x, offset_y, oc_r = sfr_funcion.pointing_oc_et(points_xy, center_xy)
        
        # Rotation
        rotation_mean, rotation_a, rotation_b, rotation_c, rotation_d, = sfr_funcion.rotation_cv(point_a, point_b, point_c, point_d, point_e)
        
        # Tilt
        pan, tilt = sfr_funcion.tilt_cv(offset_x, offset_y, self.cfg.pixel_size, self.cfg.efl)
        
        # FOV
        fov_d, fov_v, fov_h = sfr_funcion.fov_Cv(point_a, point_b, point_c, point_d, point_g, point_h, point_j, point_k, 
           self.cfg.image_circle, self.cfg.distD_design_percent, self.cfg.distV_design_percent, self.cfg.distH_design_percent, self.cfg.fov_design)
        
        # Brightness
        brightness = sfr_funcion.brightness_cv(image, self.cfg.brightness_center, self.cfg.brightness_roi_size)
        brightness_a, brightness_b, brightness_c, brightness_d = brightness
        avg_brightness = (brightness_a + brightness_b + brightness_c + brightness_d) * 0.25
        mtf = {}
        f = 0
        for idx in range(mtf_data.shape[0]):
            if idx >= 4:
                f = 3
            if idx >= 12:
                f = 6
            if idx >= 20:
                f = 8
            mtf[f'SFR_40cm_Ny2_0{f}0F_ROI{idx+1}'] = 100 * mtf_data[idx, 0]
            
        data = {
                f'SFR_40cm_Ny2_000F_Min': sfr00f_min,
                f'SFR_40cm_Ny2_030F_Min': sfr03f_min,
                f'SFR_40cm_Ny2_060F_Min': sfr06f_min,
                f'SFR_40cm_Ny2_080F_Min': sfr08f_min,
                f'SFR_40cm_Ny2_060F_Balance': sfr06f_balance,
                f'SFR_40cm_Ny2_080F_Balance': sfr08f_balance,
                
                f'SFR_40cm_Brightness_A': brightness_a,
                f'SFR_40cm_Brightness_B': brightness_b,
                f'SFR_40cm_Brightness_C': brightness_c,
                f'SFR_40cm_Brightness_D': brightness_d,
                f'SFR_40cm_Brightness_Brightness_Mean': avg_brightness,

                f'SFR_Ax': point_a[0],
                f'SFR_Ay': point_a[1],
                f'SFR_Bx': point_b[0],
                f'SFR_By': point_b[1],
                f'SFR_Cx': point_c[0],
                f'SFR_Cy': point_c[1],
                f'SFR_Dx': point_d[0],
                f'SFR_Dy': point_d[1],
                f'SFR_Ex': point_e[0],
                f'SFR_Ey': point_e[1],
                
                f'SFR_Center_X': oc_x,
                f'SFR_Center_Y': oc_y,
                f'SFR_Pointing_X': offset_x,
                f'SFR_Pointing_Y': offset_y,
                
                f'SFR_Pan': pan,
                f'SFR_Tilt': tilt,
                
                f'SFR_Rotation_A': rotation_a,
                f'SFR_Rotation_B': rotation_b,
                f'SFR_Rotation_C': rotation_c,
                f'SFR_Rotation_D': rotation_d,
                f'SFR_Rotation_Mean': rotation_mean,
                
                f'SFR_FOV_D': fov_d,
                f'SFR_FOV_H': fov_h,
                f'SFR_FOV_V': fov_v,
        }
        data = {**mtf, **data}
        data = {k: round(v, 5) for k, v in data.items()}
        device_id = utils.GlobalConfig.get_device_id()
        os.makedirs(save_path, exist_ok=True)
        save_file_name = os.path.join(save_path, 'sfr.csv')
        utils.save_dict_to_csv(data, str(save_file_name))
        
        
        
def find_first_raw_file(root_folder):
    results = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if os.path.basename(dirpath) == "sfr_files":
            raw_files = sorted(f for f in filenames if f.endswith(".raw"))
            if raw_files:
                full_path = os.path.join(dirpath, raw_files[0])
                results[dirpath] = full_path
    return results
        
    
if __name__ == '__main__':
    file_name = r'E:\Wrok\ERS\Diamond CV\Module Images'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_cv.yaml'
    sfr = CalSFR(config_path)
    # sfr.func(file_name, save_path)
    raw_file_dict = find_first_raw_file(file_name)
    for folder, file in raw_file_dict.items():
        sfr.func(file, save_path)
    # utils.process_files(file_name, sfr.func, '.raw', save_path)
    print('sfr finished!') 
