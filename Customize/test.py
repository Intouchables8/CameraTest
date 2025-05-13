import os
import sys
# from scipy.ndimage import median_filter
# from numba import jit, prange
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
import cv2
from Common import edge_median_filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as mplstyle
matplotlib.use('agg')
mplstyle.use('fast')
plt.ioff()
from scipy.optimize import curve_fit
from Common import utils
from Common.utils import time_it_avg, time_block

class TVEBlemish:
    def __init__(self, config_path):
        cfg = utils.load_config(config_path).light
        self.image_cfg = cfg.image_info
        self.cfg = cfg.blemish
        self.median_kern =TVEBlemish._generate_median_kern(self.cfg.kernel_size)
        self.offsets, self.pad_width, self.kernel_size = TVEBlemish._pre_compute_kernel(self.median_kern)
    @staticmethod
    def _pre_compute_kernel(kernel):
        kernel_height = kernel.shape[0]
        pad = kernel_height // 2
        # 找到 kernel 中值为 1 的边缘像素索引
        edge_indices = np.where(kernel == 1)
        # 预计算偏移量：行索引 * kernel_size + 列索引
        offsets = edge_indices[0] * kernel_height + edge_indices[1]
        return offsets, pad, kernel_height

    @staticmethod
    def _optical_center(src_image, thresh=0.9):
        image = src_image.copy()
        total_pixel = image.size
        pixel_thresh = thresh * total_pixel
        gray_val, counts = np.unique(image, return_counts=True) 
        gray_thresh = gray_val[np.searchsorted(np.cumsum(counts), pixel_thresh)]
        mask = src_image > gray_thresh
        
        low = np.sum(src_image[mask])
        x_u = np.arange(0.5, src_image.shape[1])
        y_u = np.arange(0.5, src_image.shape[0])
        oc_x = np.sum(x_u * np.sum(src_image * mask, axis=0)) / low
        oc_y = np.sum(y_u * np.sum(src_image * mask, axis=1)) / low
        return oc_x, oc_y

    @staticmethod
    def _fit_image(image, oc_x, oc_y, pad_width, poly_degree):
        image = image.astype(np.float32)
        rows, cols = image.shape
        y, x = np.ogrid[:rows, :cols]
        distances = np.sqrt((x - oc_x) ** 2 + (y - oc_y) ** 2).ravel()

        # 3 次多项式拟合（避免 6 次的震荡）
        coefficients = np.polyfit(distances.flatten(), image.flatten(), poly_degree)
        poly = np.poly1d(coefficients)

        # 创建填充后的图像
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        rows_pad, cols_pad = padded_image.shape
        
        # 计算填充区域的距离（相对于新的中心）
        y_pad, x_pad = np.indices((rows_pad, cols_pad))  
        distance_pad = np.sqrt((x_pad - (oc_x + pad_width)) ** 2 + (y_pad - (oc_y + pad_width)) ** 2)

        # 计算填充值
        valuePad = poly(distance_pad)
        
        # 修正填充区域
        padded_image[:pad_width, :] = valuePad[:pad_width, :]  # 上边界
        padded_image[-pad_width:, :] = valuePad[-pad_width:, :]  # 下边界
        padded_image[:, :pad_width] = valuePad[:, :pad_width]  # 左边界
        padded_image[:, -pad_width:] = valuePad[:, -pad_width:]  # 右边界

        return np.clip(np.round(padded_image), 0, 1023).astype(np.uint16)

    @staticmethod
    def _generate_median_kern(kernel_size):
        median_kern = np.ones(kernel_size, np.uint8)
        median_kern[1:-1, 1:-1] = 0
        return median_kern

    @staticmethod
    def _generate_masks(image_size, side_mask_dist, mask_radius, pad_width):
        h, w = image_size
        mask = utils.generate_mask(image_size, mask_radius)
        # side suppression
        suppression_mask = np.ones(image_size) * 0.5
        suppression_mask[pad_width: h - pad_width, pad_width: w - pad_width] = 1
        # side mask
        side_mask = np.zeros(image_size, np.uint8)
        side_mask[side_mask_dist: h-side_mask_dist, side_mask_dist: w-side_mask_dist] = 1
        total_mask = np.bitwise_and(mask, side_mask > 0)
        return mask, total_mask, side_mask, suppression_mask

    @staticmethod
    def _morph_operation(image, morph_kern):
        kern = np.ones(morph_kern)
        imageMorped = cv2.morphologyEx(image, cv2.MORPH_OPEN, kern)
        return imageMorped

    @staticmethod
    def _add_mask(image, mask):
        image[mask == 0] = 0
        return image
    
    @staticmethod
    def _get_thresh(datas):
        hist, bin_edges = np.histogram(datas, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
        def gaussian(x, a, mean, sigma):
            x = np.asarray(x) 
            return a * np.exp(-(x - mean)**2 / (2 * sigma**2))
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[datas.max(), datas.mean(), datas.std()], maxfev = 2000)
        _, mean_opt, sigma_opt = popt
        return mean_opt, sigma_opt

    @staticmethod
    def _get_blemish_area(blemish_map, down_scale_factor, min_area):
        area = 0
        blemish_count = 0
        contours, _ = cv2.findContours(blemish_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            curArea = cv2.contourArea(contours[i])
            if curArea > min_area:
                area += cv2.contourArea(contours[i])
                blemish_count += 1
        blemish_area = down_scale_factor ** 2 * area
        return blemish_area, blemish_count
    
    @staticmethod
    def _visualization(image, blemish_map, min_area, blemish_count, final_thresh, blemish_area, down_scale_factor, delta_image, save_path, dpi=50):
        image_8bit = (image >> 2).astype(np.uint8)
        I_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(blemish_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:  # 过滤小面积区域
                x, y, w, h = cv2.boundingRect(cnt)  # 获取矩形 (x, y, w, h)
                cv2.rectangle(I_rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)  # 绘制矩形

        device_id = utils.GlobalConfig.get_device_id()
        data = {    
                'Device ID': str(device_id),
                'Thresh': str(final_thresh),
                'FF_Blemish_Ct': str(blemish_count),
                'FF_Blemish_Area': str(blemish_area)
                }
        save_file_path = os.path.join(save_path, 'blemish_data.csv')
        utils.save_dict_to_csv(data, str(save_file_path))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)  # 设置较高 dpi 提高保存质量
        fig.suptitle(device_id)

        # 子图 1: 原始图像
        axes[0, 0].imshow(image_8bit, cmap='gray')
        axes[0, 0].set_title(f'{down_scale_factor}X down scaled image')
        axes[0, 0].axis("off")  # 移除坐标轴

        # 子图 2: Delta with mask
        axes[0, 1].imshow(delta_image, cmap='gray')
        axes[0, 1].set_title('Delta with mask')
        axes[0, 1].axis("off")

        # 子图 3: 处理后的 binary blemish map
        axes[1, 0].imshow(blemish_map, cmap='gray')
        axes[1, 0].set_title(f'bwBlemish, Thresh:{final_thresh:.3f}')
        axes[1, 0].axis("off")

        # 子图 4: 画框后的最终图像
        axes[1, 1].imshow(I_rgb)
        axes[1, 1].set_title(f'Cnt:{blemish_count}, Area:{blemish_area}')
        axes[1, 1].axis("off")

        # 自动调整子图布局
        plt.tight_layout()
        savepath = os.path.join(save_path, f'{"Pass" if blemish_count == 0 else "Fail"}_{device_id}.jpg')
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.1, dpi=dpi)  # 设置 dpi 提高保存质量
        plt.close(fig)  # 释放内存
    
    # @time_it_avg(2)   
    def run(self, image, save_path):
        
        # -------------- Down Scale --------------
        if not self.cfg.down_scale_factor == 1:
            image = cv2.resize(image,(image.shape[1] //self.cfg.down_scale_factor, image.shape[0] //self.cfg.down_scale_factor))

        _, bw_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        # erode_bw = cv2.erode(bw_image, np.ones([15, 15]))
        conts, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_cont = max(conts, key = cv2.contourArea)  # key为指定函数， 面积函数
        # 使用凸包 获得图像边缘位置 找到掩膜
        hull = cv2.convexHull(max_cont)
        mask_tmp = np.zeros_like(image, np.uint8)
        mask_tmp = cv2.drawContours(mask_tmp, [hull], 0 , 255, -1)
        mask = cv2.erode(mask_tmp, np.ones([15, 15]))  # 防止出现外围一圈白线
        image[mask==0] = 0
        
        mask = cv2.erode(mask, np.ones([75, 75]))
        self.mask = mask
        
        # -------------- Denoise --------------
        image_denoise = cv2.GaussianBlur(image, (0, 0), self.cfg.denoise_sigma) 
        image_median = edge_median_filter.edge_median_filter(image_denoise.astype(np.float64), self.offsets, self.pad_width)
        delta_image = image_median.astype(np.float32) - image_denoise.astype(np.float32)
        mask_delta_image = TVEBlemish._add_mask(delta_image, self.mask)

        # -------------- Thresh ------------
        datas = np.compress(self.mask.ravel(), mask_delta_image.ravel())
        mean, sigma = TVEBlemish._get_thresh(datas)
        adapt_thresh = mean + 4 * sigma
        final_thresh = np.clip(adapt_thresh, self.cfg.threshold_min_max[0], self.cfg.threshold_min_max[1])
        final_thresh = 1.0
        # ------------- Blemish Map ----------
        # 计算阈值条件
        condition = mask_delta_image >= final_thresh if self.cfg.seg_dir == 1 else np.abs(mask_delta_image) >= final_thresh
        blemish_map = np.where(condition, 255, 0).astype(np.uint8)
        
        # ------------- morph ---------------
        if self.cfg.morph:
            blemish_map = TVEBlemish._morph_operation(blemish_map, self.cfg.morph_kern)

        # ------------- Blemish Area --------
        blemish_area, blemish_count = TVEBlemish._get_blemish_area(blemish_map, self.cfg.down_scale_factor, self.cfg.min_area)
        
        # ------------- Visualization --------
        if self.cfg.debug_flag and blemish_count > 0:
            
            os.makedirs(save_path, exist_ok=True)
            TVEBlemish._visualization(image, blemish_map, self.cfg.min_area, blemish_count, final_thresh, blemish_area, self.cfg.down_scale_factor, delta_image, save_path, self.cfg.dpi)

        data = {
                'Blemish_Count': blemish_count,
                'Blemish_area': blemish_area,
                'Thresh': final_thresh,
                'Blemish_Count': blemish_count, 
        }
        return data
    
    def func(self, file_name, save_path):
        image = utils.load_image(file_name, self.image_cfg.image_type, self.image_cfg.image_size, self.image_cfg.crop_tblr)
        if self.cfg.sub_black_level:
            image = utils.sub_black_level(image,  self.cfg.black_level)

        if self.cfg.input_pattern == 'Y' and self.image_cfg.bayer_pattern != 'Y':
            image = utils.bayer_2_y(image, self.image_cfg.bayer_pattern)
        self.run(image, save_path)
    
if __name__ == '__main__':
    file_name = r'E:\Wrok\Temp\20250424\limit_mura\Main_W127_1.bmp'
    save_path = r'E:\Wrok\Temp\20250424\limit_mura'
    config_path = r'D:\Code\CameraTest\Config\config_test.yaml'
    blemish = TVEBlemish(config_path)
    blemish.func(file_name, save_path)
    # utils.process_files(file_name, blemish.func, '.raw', save_path)
    print('blemish finish')


    
    
