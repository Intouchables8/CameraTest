import numpy as np
import cv2
from pathlib import Path
import sys
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils
# from concurrent.futures import ThreadPoolExecutor
def _debug_image(valid_row_index, valid_col_index, row_avg, col_avg, row_diff, col_diff, channel, save_path):
    utils.FilePath.create_folder(save_path)
    save_file_path = save_path / (utils.GlobalConfig.get_device_id() + '_DRCL.csv') 
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
    utils.save_lists_to_csv(data, name, save_file_path)

def drcl(src, thresh, mask_radius, min_pixel, neighbor, debug_flag, save_path, channel):
    image = src.copy()
    image_size = image.shape
    if mask_radius > 0:
        mask = utils.generate_mask(image_size, mask_radius)
        image[mask == 0] = 0
        row_sum = image.sum(axis=1)
        row_count = np.count_nonzero(mask, axis=1)
        row_avg = row_sum / row_count

        col_sum = image.sum(axis=0)
        col_count = np.count_nonzero(mask, axis=0)
        col_avg = col_sum / col_count
    
    else:
        row_avg = image.mean(axis = 1)
        col_avg = image.mean(axis = 0)
        row_count, col_count = image_size

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
    row_index_index = np.where(row_count[row_index] > min_pixel)
    col_index_index = np.where(col_count[col_index] > min_pixel)
    valid_row_index = row_index[row_index_index]
    valid_col_index = col_index[col_index_index]
    
    if debug_flag:
        if len(valid_row_index) > 0 or len(valid_col_index):
            _debug_image(valid_row_index, valid_col_index, row_avg, col_avg, row_diff, col_diff, channel, save_path)
    return valid_row_index, valid_col_index

def defect_row_col_light(image, thresh, input_pattern, mask_radius, min_pixel, neighbor, save_path, csv_output, debug_flag):
    save_path = Path(save_path)
    if input_pattern != 'Y':
        pass
    
    else:
        row_index, col_index = drcl(image, thresh, mask_radius, min_pixel, neighbor, debug_flag, save_path, 'Y')
        data = {
            'DRCL_Thresh': str(thresh),
            'Defect_Row_Light': str(len(row_index)),
            'Defect_Col_Light': str(len(col_index))
        }
    if csv_output:
        save_file_path = save_path / 'drcl_data.csv'
        utils.save_dict_to_csv(data, save_file_path) 
        
    
    

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
    cfg = utils.load_config(config_path).light
    image_cfg = cfg.image_info
    cfg = cfg.defect_row_col_light
    image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if cfg.sub_black_level:
        image = utils.sub_black_level(image, image_cfg.black_level)
        
    if cfg.input_pattern == 'Y' and image_cfg.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, image_cfg.bayer_pattern)
        
    defect_row_col_light(image, cfg.thresh, cfg.input_pattern, cfg.mask_radius, cfg.min_pixel, cfg.neighbor, save_path, cfg.csv_output, cfg.debug_flag)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\ET\light.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_et.yaml'
    func(file_name, save_path, config_path)
    
    print('drcl finished!')
