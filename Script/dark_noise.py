import numpy as np
import sys
from pathlib import Path
ROOTPATH = Path(__file__).parent.parent
sys.path.append(str(ROOTPATH))
from Common import utils
from pathlib import Path
def dark_noise(images, csv_output, save_path):
    save_path = Path(save_path)
    # ------------------------ Tempo Noise ------------------------
    # 计算每个像素在所有帧上的均值（保持维度便于后续广播）
    # tempo_pixel: 对每个像素沿帧计算样本标准差，然后取平均
    tempo_pixel = np.std(images, axis=2, ddof=1).mean()

    # tempo_row: 先计算每行每帧的均值，shape 为 (row, channel)
    row_means = images.mean(axis=1)
    # 每行的样本标准差（沿通道维度），再对所有行取平均
    tempo_row = np.std(row_means, axis=1, ddof=1).mean()

    # tempo_col: 先计算每列每帧的均值，shape 为 (col, channel)
    col_means = images.mean(axis=0)
    # 每列的样本标准差（沿通道维度），再对所有列取平均
    tempo_col = np.std(col_means, axis=1, ddof=1).mean()

    # 总的 Tempo Total
    tempo_total = np.sqrt(tempo_pixel**2 + tempo_row**2 + tempo_col**2)

    # Dark_Noise_Total: 整个图像的样本标准差
    dark_noise_total = np.std(images, ddof=1)

    # ------------------------ Fixed Pattern Noise (FPN) ------------------------
    p_l = images.mean(axis=2)           
    # 每个像素在所有帧上的均值
    # dark_noise_fpn_total: 每个像素均值与总体均值（P_total/(L*N)）之间的总体标准差
    dark_noise_fpn_total = np.std(p_l, ddof=0)  # 这里用总体标准差

    # fpn_row: 计算每行均值的相邻差分（循环处理），并归一化
    h_n = p_l.mean(axis=1)
    fpn_row = np.sqrt(np.mean(np.square((h_n - np.roll(h_n, 1)) / np.sqrt(2))))

    # fpn_col: 同理，计算每列均值的相邻差分（循环处理）
    v_n = p_l.mean(axis=0)
    fpn_col = np.sqrt(np.mean(np.square((v_n - np.roll(v_n, 1)) / np.sqrt(2))))

    # fpn_pixel: 根据总 FPN 和行、列分量求解像素级 FPN
    fpn_pixel = np.sqrt(dark_noise_fpn_total**2 - fpn_row**2 - fpn_col**2)

    # Ratios
    ratio_rfpn = tempo_total / fpn_row
    ratio_cfpn = tempo_total / fpn_col
    ratio_pfpn = tempo_total / fpn_pixel

    noise_data = {
                'Dark_Noise_Total': str(dark_noise_total),
                'dark_noise_fpn_total': str(dark_noise_fpn_total),
                'Dark_Noise_FPN_Row': str(fpn_row),
                'Dark_Noise_FPN_Col': str(fpn_col),
                'Dark_Noise_FPN_Pixel': str(fpn_pixel),
                'Dark_Noise_tempo_total': str(tempo_total),
                'Dark_Noise_tempo_row': str(tempo_row),
                'Dark_Noise_tempo_col': str(tempo_col),
                'Dark_Noise_tempo_pixel': str(tempo_pixel),
                'Dark_Noise_Ratio_rFPN': str(ratio_rfpn),
                'Dark_Noise_Ratio_cFPN': str(ratio_cfpn),
                'Dark_Noise_Ratio_pFPN': str(ratio_pfpn),
                
    }

    if csv_output:
        save_file_path = save_path / 'dark_noise_data.csv'
        utils.save_dict_to_csv(noise_data, save_file_path)

def func(file_name, save_path, config_path):
    config_path = Path(config_path)
    cfg = utils.load_config(config_path).dark
    image_cfg = cfg.image_info
    noise_cfg = cfg.dark_noise
    images = utils.load_images(file_name, noise_cfg.image_count, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if noise_cfg.sub_black_level:
        images = utils.sub_black_level(images, image_cfg.black_level)
    
    dark_noise(images, noise_cfg.csv_output, save_path)
    return True

if __name__ == '__main__':
    file_name = r'E:\Wrok\Temp\CaliforniaFATP\20250312\20250311\offline\holder_1\356YW33GB6001T\20250311171830\Dark\noise\camera\result\frame_9.raw'
    save_path = r'E:\Wrok\Temp\CaliforniaFATP\20250312\20250311\offline\holder_1\356YW33GB6001T\20250311171830\Dark\noise\camera\result'
    config_path = r'G:\CameraTest\Config\config_california.yaml'
    import time 
    start = time.time()
    func(file_name, save_path, config_path)
    print(time.time() - start)
    print('dark noise finished!')