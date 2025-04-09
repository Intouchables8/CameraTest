from Common import utils
from Project.CV import sfr
from Customize import sfr_funcion
import numpy as np
class CalSFR:
    def __init__(self, config_path):
        self.sfr = sfr.SFR(config_path, 1)
    
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.sfr.image_tpye, self.sfr.image_size, self.sfr.crop_tblr)
        if self.sfr.sub_black_level:
            image = utils.sub_black_level(image, self.sfr.black_level)
        
        if self.sfr.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.sfr.sfr.bayer_pattern)
        
        if image.dtype == np.uint16:
            image = (image >> 2).astype(np.uint8)
        
        # 定位block
        block_roi_center_xy, _, _, bw_image = self.sfr.locate_block_cv(image, save_path)
        
        # 定位所有point相关坐标
        center_xy = (self.sfr.image_size[1] // 2, self.sfr.image_size[0] // 2)
        _, centroid =self.sfr.find_connected_area(bw_image, self.sfr.point_thresh, 'Point')
        index = np.argmin(utils.calcu_distance(centroid, center_xy))
        chart_center_xy = centroid[index]
        points_xy = sfr_funcion.select_point_cv(centroid, chart_center_xy, self.sfr.n_point, self.sfr.point_dist_from_center, self.sfr.clockwise, self.sfr.debug_flag)
    
        # 选择roi
        all_roi_center_xy = self.sfr.select_roi(block_roi_center_xy, self.sfr.roi_index)
        
        # 创建rect
        all_roi_rect = self.sfr.get_roi_rect(image, all_roi_center_xy)
            
        # 计算mtf
        self.sfr.calcu_mtf(image, all_roi_rect, save_path)
        
        
# def find_first_raw_file(root_folder):
#     results = {}
#     # 遍历主文件夹下的所有子文件夹
#     for subfolder in root_path.glob("**/sfr_files"):  # 查找所有名为 'sfr' 的文件夹
#         if subfolder.is_dir():  # 确保是目录
#             raw_files = sorted(subfolder.glob("*.raw"))  # 获取 .raw 文件，并排序
#             if raw_files:  # 确保存在 .raw 文件
#                 results[subfolder] = raw_files[0]  # 记录第一个 .raw 文件路径

#     return results
        
    
if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\CV\sfr.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_cv.yaml'
    sfr = CalSFR(config_path)
    sfr.func(file_name, save_path)
    # raw_file_dict = find_first_raw_file(file_name)
    # for folder, file in raw_file_dict.items():
    #     sfr.func(file, save_path)
    # utils.process_file_or_folder(file_name, '.raw', sfr.func, save_path)
    print('sfr finished!') 
