import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils



def color_calibration(file_name, image_size, image_type, crop_tblr, bayer_pattern,color_temperture, sub_black_level, black_level, csv_output, save_path=None):
    image = utils.load_image(str(file_name), image_type, image_size, crop_tblr)
    r, gr, gb, b = utils.split_channel(image, bayer_pattern)
    rows, cols = r.shape
    cx, cy = rows // 2, cols // 2
    roi_height, roi_width = rows // 8, cols // 8
    half_height, half_width = roi_height // 2, roi_width // 2
    row_start = cy - half_height
    row_end = row_start + roi_height
    col_start = cx - half_width
    col_end = col_start + roi_width
    
    r_avg = r[row_start: row_end, col_start: col_end].mean()
    gr_avg = gr[row_start: row_end, col_start: col_end].mean()
    gb_avg = gb[row_start: row_end, col_start: col_end].mean()
    b_avg = b[row_start: row_end, col_start: col_end].mean()
    
    if sub_black_level:
        r_avg -= black_level
        gr_avg -= black_level
        gb_avg -= black_level
        b_avg -= black_level
    
    G = (gr_avg + gb_avg) * 0.5
    R_G = r_avg / G
    B_G = b_avg / G
    Gb_Gr = gb_avg / gr_avg
    
    data = {
                f'{color_temperture}K_CC_R/G': R_G,
                f'{color_temperture}K_CC_B/G': B_G,
                f'{color_temperture}K_CC_Gb/Gr': Gb_Gr
            }
    
    if csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'cc_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))

def func(file_name, save_path, config_path, color_temperture):
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    cc_cfg = cfg.color_calibration
    color_calibration(file_name, image_cfg.image_size, image_cfg.image_type, image_cfg.crop_tblr, image_cfg.bayer_pattern, color_temperture, cc_cfg.sub_black_level, cc_cfg.black_level, cc_cfg.csv_output, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\RGB\light\20241221_151813__0_AS_DNPVerify_377TT04G9L0188.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_rgb.yaml'
    color_temperture = '2800'
    func(file_name, save_path, config_path, color_temperture)
    print('CC finished!')