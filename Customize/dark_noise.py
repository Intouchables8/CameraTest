import numpy as np
import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils
import os
def calcu_noise(images, channel=''):
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

    data = {
                f'Dark_Noise_{channel}Total': str(dark_noise_total),
                f'dark_noise_{channel}FPN_total': str(dark_noise_fpn_total),
                f'Dark_Noise_{channel}FPN_Row': str(fpn_row),
                f'Dark_Noise_{channel}FPN_Col': str(fpn_col),
                f'Dark_Noise_{channel}FPN_Pixel': str(fpn_pixel),
                f'Dark_Noise_{channel}tempo_total': str(tempo_total),
                f'Dark_Noise_{channel}tempo_row': str(tempo_row),
                f'Dark_Noise_{channel}tempo_col': str(tempo_col),
                f'Dark_Noise_{channel}tempo_pixel': str(tempo_pixel),
                f'Dark_Noise_{channel}Ratio_rFPN': str(ratio_rfpn),
                f'Dark_Noise_{channel}Ratio_cFPN': str(ratio_cfpn),
                f'Dark_Noise_{channel}Ratio_pFPN': str(ratio_pfpn),
                
    }
    return data

def dark_noise(images, input_pattern, csv_output, save_path):
    if input_pattern == 'Y':
        data = calcu_noise(images)
    else:
        r, gr, gb, b = utils.split_channel(images, input_pattern)
        r_data = calcu_noise(r, 'R_')
        gr_data = calcu_noise(gr, 'Gr_')
        gb_data = calcu_noise(gb, 'Gb_')
        b_data = calcu_noise(b, 'B_')
        data = {**r_data, **gr_data, **gb_data, **b_data}
    
    if csv_output:
        os.makedirs(save_path, exist_ok=True)
        save_file_path = os.path.join(save_path, 'dark_noise_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))
    return data

def func(file_name, save_path, config_path):
    cfg = utils.load_config(config_path).dark
    image_cfg = cfg.image_info
    noise_cfg = cfg.dark_noise
    images = utils.load_images(file_name, noise_cfg.image_count, image_cfg.image_type, image_cfg.image_size, image_cfg.crop_tblr)

    if noise_cfg.sub_black_level:
        images = utils.sub_black_level(images, image_cfg.black_level)
    
    dark_noise(images, noise_cfg.input_pattern, noise_cfg.csv_output, save_path)
    return True

if __name__ == '__main__':
    # file_name = r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\01\377TT04G9L000Z_Dark_0.raw'
    save_path = r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark'
    config_path = r'D:\Code\CameraTest\Config\config_rgb.yaml'
    files = [r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\01\377TT04G9L000Z_Dark_0.raw', 
            r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\02\377TT04G9L00B4_Dark_0.raw',
            r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\03\377TT04G9L01M7_Dark_0.raw',
            r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\04\377TT04G9L024C_Dark_0.raw',
            r'C:\Users\wangjianan\Desktop\Innorev_Result\Dark\imaegs\05\377TT04G9L028D_Dark_0.raw']
    for file_name in files:
        func(file_name, save_path, config_path)
        print('dark noise finished!')