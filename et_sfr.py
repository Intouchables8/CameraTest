from Common import utils
from Project.ET import sfr
import numpy as np
class CalSFR:
    def __init__(self, config_path):
        self.sfr = sfr.SFR(config_path)
    
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.sfr.image_tpye, self.sfr.image_size, self.sfr.crop_tblr)
        if self.sfr.sub_black_level:
            image = utils.sub_black_level(image, self.sfr.black_level)
        
        if self.sfr.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.sfr.sfr.bayer_pattern)
        
        if image.dtype == np.uint16:
            image = (image >> 2).astype(np.uint8)
        
        # 定位block
        block_roi_center_xy, block_centroid, inner_block_center_xy, points_xy = self.sfr.localte_block_california(image, str(save_path))
        
        # 选择roi
        all_roi_center_xy = self.sfr.select_roi(block_roi_center_xy, self.sfr.roi_index)
        
        # 创建rect
        all_roi_rect = self.sfr.get_roi_rect(image, all_roi_center_xy)
            
        # 计算mtf
        self.sfr.calcu_mtf(image, all_roi_rect, str(save_path))
        
        
if __name__ == '__main__':
    file_name = r'.\image\ET\sfr.raw'
    save_path = r'.\result'
    config_path = r'G:\CameraTest\Config\config_et.yaml'
    sfr = CalSFR(config_path)
    sfr.func(file_name, save_path)
    # utils.process_file_or_folder(file_name, '.raw', sfr.func, save_path)
    # utils.process_file_or_folder(file_name, '.raw', sfr.func, save_path)
    print('sfr finished!') 
