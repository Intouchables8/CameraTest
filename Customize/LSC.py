import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils

#  需要提供具体的操作细节
def lsc(file_name, image_size, image_type, crop_tblr, bayer_pattern,color_temperture, sub_black_level, black_level, csv_output, save_path=None):
 
    
    data = {

            }
    
    if csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'cc_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))

def func(file_name, save_path, config_path, color_temperture):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    cc_cfg = cfg.LSC
    lsc(file_name, image_cfg.image_size, image_cfg.image_type, image_cfg.crop_tblr, image_cfg.bayer_pattern, color_temperture, cc_cfg.sub_black_level, cc_cfg.black_level, cc_cfg.csv_output, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\RGB\light\20241221_151813__0_AS_DNPVerify_377TT04G9L0188.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_rgb.yaml'
    color_temperture = '2800'
    func(file_name, save_path, config_path, color_temperture)
    print('CC finished!')