import numpy as np
import cv2
import os
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(str(ROOTPATH))
from Common import utils
# from concurrent.futures import ThreadPoolExecutor
def _debug_image(valid_row_index, valid_col_index, row_avg, col_avg, row_diff, col_diff, channel, save_path):
    
    channel_data = []
    item_data = []
    index_data = []
    avg_data =[]
    diff_data =[]

    name = ['channel', 'item ','index', 'avg', 'diff']
    if len(valid_row_index) > 0:
        for index in valid_row_index:
            channel_data.append(channel)
            item_data.append('row')
            index_data.append(index)
            avg_data.append(row_avg[index])
            diff_data.append(row_diff[index])
    if len(valid_col_index) > 0:   
        for index in valid_col_index:
            channel_data.append(channel)
            item_data.append('col')
            index_data.append(index)
            avg_data.append(col_avg[index])
            diff_data.append(col_diff[index])
        
    data = zip(channel_data, item_data, index_data, avg_data, diff_data)
    save_file_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_DRCD.csv'))
    utils.save_lists_to_csv(data, name, str(save_file_path))

def drcd(image, thresh, neighbor, debug_flag, save_path, channel):
    row_avg = image.mean(axis = 1)
    col_avg = image.mean(axis = 0)

    if neighbor > 1:
        kern = np.array((1, 0, 1)) if neighbor == 1 else np.array((1, 1, 0, 1, 1))
        # nRow = cv2.filter2D(np.ones((rows,1)), -1, kern, borderType=cv2.BORDER_REFLECT101)
        # nCol = cv2.filter2D(np.ones((cols,1)), -1, kern, borderType=cv2.BORDER_REFLECT101)

        row_neighb_total = cv2.filter2D(row_avg, -1, kern, borderType=cv2.BORDER_REFLECT101)
        row_neighb_avg = row_neighb_total / (2 * neighbor)
        row_diff = row_avg - row_neighb_avg.reshape(-1)

        col_neighb_total = cv2.filter2D(col_avg, -1, kern, borderType=cv2.BORDER_REFLECT101)
        col_neighb_avg = col_neighb_total / (2 * neighbor)
        col_diff = col_avg - col_neighb_avg.reshape(-1)

    else:
        row_diff = np.abs(row_avg[:-1] - row_avg[1:])
        col_diff = np.abs(col_avg[:-1] - col_avg[1:])
    
    ## Threshold
    row_index = np.where(row_diff > thresh)[0]
    col_index = np.where(col_diff > thresh)[0]

    if debug_flag:
        if len(row_index) > 0 or len(col_index):
            
            os.makedirs(save_path, exist_ok=True)
            _debug_image(row_index, col_index, row_avg, col_avg, row_diff, col_diff, channel, save_path)
    return row_index, col_index

def defect_row_col_dark(image, thresh, input_pattern, neighbor, save_path, csv_output, debug_flag):
    if input_pattern != 'Y':
        pass
    
    else:
        row_index, col_index = drcd(image, thresh, neighbor, debug_flag, save_path, 'Y')
        data = {
            'DRCD_Thresh': thresh,
            'Defect_Row_Dark': str(len(row_index)),
            'Defect_Col_Dark': str(len(col_index))
        }
    if csv_output:
        
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'drcd_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path)) 
    return data 
        
    
def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).dark
    image_cfg = cfg.image_info
    cfg = cfg.defect_row_col_dark
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
        
    if cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)
        
    defect_row_col_dark(image, cfg.thresh, cfg.input_pattern, cfg.neighbor, save_path, cfg.csv_output, cfg.debug_flag)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\CV\dark\Ketron_P0C_FF2_Line1_DARK1_EOL-Dark_373KQ11GC300V8_030703111601010e0b0300001a08_20241228153651_0.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_cv.yaml'
    func(file_name, save_path, config_path)
    
    print('drcd finished!')
