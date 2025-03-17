import numpy as np
import sys
from pathlib import Path
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils
from pathlib import Path
import cv2


def dsnu(image, roi_size, except_dpd=False, dpd_map=None, thresh=0, csv_output=False, save_path=None):
    save_path = Path(save_path)
    if except_dpd:
        avg = image.mean()
        image[dpd_map > 0] = avg + thresh

    integralImage = cv2.integral(image)  # 对图像的每个点与之前的所有点进行求和
    roi_size_height = roi_size[0]
    rows, cols = image.shape
    W, H = image.shape[1], image.shape[0]
    col = 0
    avg_block = []
    for x in range(0, W, roi_size_height):
        row = 0
        for y in range(0, H, roi_size_height):
            br = (min(y + roi_size_height, H), min(x + roi_size_height, W))  # 右下角
            tl = (y, x)  # 左上角
            tr = (y, min(x + roi_size_height, W))  # 右上角
            bl = (min(y + roi_size_height, H), x)  # 左下角
            sumBlock = integralImage[br] - integralImage[tr] - integralImage[bl] + integralImage[tl]
            avgBlock = sumBlock / ((br[0] - tl[0]) * (br[1] - tl[1]))
            avg_block.append(avgBlock)
            row += 1
        col += 1
    max_dsnu = max(avg_block)
    min_dsnu = min(avg_block)
    DSNU = max_dsnu - min_dsnu
    data = {
            'DSNU': DSNU,
            'Max_DSNU': max_dsnu,
            'MIN_DSNU': min_dsnu,
    }

    
    if csv_output:
        save_file_path = save_path / 'dsnu_data.csv'
        
        utils.save_dict_to_csv(data, save_file_path)

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    file_name = Path(file_name)
    cfg = utils.load_config(config_path).dark
    image_cfg = cfg.image_info
    dsnu_cfg = cfg.DSNU
    thresh = cfg.defect_pixel_dark.thresh
    if dsnu_cfg.image_count > 1:
        images = utils.load_images(file_name, dsnu_cfg.image_count, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)
        image = images.mean(axis=2)
    else:
        image = utils.load_image(file_name, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)
        
    dsnu(image, dsnu_cfg.roi_size, None, thresh, dsnu_cfg.csv_output, dsnu_cfg.debug_flag, save_path)
    return True

if __name__ == '__main__':
    file_name = r'G:\CameraTest\image\CV\dark\Ketron_P0C_FF2_Line1_DARK1_EOL-Dark_373KQ11GC300V8_030703111601010e0b0300001a08_20241228153651_0.raw'
    save_path = r'G:\CameraTest\result'
    config_path = r'G:\CameraTest\Config\config_cv.yaml'
    import time 
    start = time.time()
    func(file_name, save_path, config_path)
    print(time.time() - start)
    print('dsnu finished!')