import numpy as np
import cv2
import os
from enum import Enum
import sys
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(str(ROOTPATH))
from Common import utils, MTF3
class BinaryMode(Enum):
    CANNY = 'CANNY'
    FIX_THRESH = 'FIX_THRESH'
    
class SFR:
    def __init__(self, config_path, visual_scalse_factor=1):
        cfg = utils.load_config(config_path).sfr
        image_cfg = cfg.image_info
        locate_cfg = cfg.locate_block
        roi_cfg = cfg.select_roi
        mtf_cfg = cfg.mtf
        
        self.cfg = cfg
        self.image_tpye = image_cfg.image_type
        self.image_size = image_cfg.image_size
        self.crop_tblr = image_cfg.crop_tblr
        self.black_level = image_cfg.black_level
        self.sub_black_level = image_cfg.sub_black_level
        self.bayer_pattern = image_cfg.bayer_pattern

        self.bw_mode = BinaryMode[locate_cfg.bw_mode]
        self.max_radius = locate_cfg.max_radius
        self.canny_thresh = locate_cfg.canny_thresh
        self.bw_thresh = locate_cfg.bw_thresh
        self.apply_hull = locate_cfg.apply_hull
        self.mask_erode_size = locate_cfg.mask_erode_size
        self.blur_size = locate_cfg.blur_size
        self.n_block = locate_cfg.n_block
        self.block_thresh = locate_cfg.block_thresh
        self.n_half_block = locate_cfg.n_half_block
        self.half_block_thresh = locate_cfg.half_block_thresh
        self.half_bloack_dist_from_center = locate_cfg.half_bloack_dist_from_center 
        self.inner_block_thresh = locate_cfg.inner_block_thresh
        self.n_point = locate_cfg.n_point
        self.point_thresh = locate_cfg.point_thresh
        self.point_dist_from_center = locate_cfg.point_dist_from_center
        self.delta_angle = locate_cfg.delta_angle
        self.debug_flag = locate_cfg.debug_flag
        self.text_size = locate_cfg.text_size
        self.thickness = locate_cfg.thickness
        self.x_offset = locate_cfg.x_offset
        self.y_offset = locate_cfg.y_offset
        self.clockwise = locate_cfg.clockwise

        self.roi_index = roi_cfg.roi_inex
        self.ny_freq = mtf_cfg.ny_freq
        self.mtf_debug = mtf_cfg.debug_flag
        self.text_value = mtf_cfg.text_value
        self.csv_output = mtf_cfg.csv_output
        
        self.rgb = None
        self.long_side = max(locate_cfg.roi_size)
        self.short_side = min(locate_cfg.roi_size)
        
        self.mtf = MTF3.MTF3(self.ny_freq)
        self.visual_scalse_factor = visual_scalse_factor
             
    def _locate_all_block(self, image, block_stats, group_sorted_index, n, min_dist):
        block_roi_center_xy = [None] * n
        block_corner__xy = [None] * n
        inner_block_center_xy = [None] * n
        i = 0
        
        for cur_group in group_sorted_index:
            if not isinstance(group_sorted_index[0], (list, np.ndarray)):
                cur_group = [cur_group]
            for index in cur_group:
                tl_x, tl_y, w, h = block_stats[index, : 4]
                cur_roi_center_xy, cur_block_corner_xy, cur_inner_center = self._get_block_coord(image[tl_y: tl_y + h, tl_x: tl_x + w], tl_x, tl_y, i, min_dist)  # R T L B
                block_corner__xy[i] = cur_block_corner_xy   # 所有框ROI 中心点的坐标
                block_roi_center_xy[i] = cur_roi_center_xy         # 所有框顶点的坐标
                inner_block_center_xy[i] = cur_inner_center
                i += 1
        return block_corner__xy, block_roi_center_xy, inner_block_center_xy
        
    def _visualize_all_points(self, block_roi_center_xy, save_path):
        index = 0
        for block in block_roi_center_xy:
            for roi_xy in block:
                x, y = roi_xy
                utils.draw_mark_with_text(self.rgb, x, y, str(index), text_size=self.text_size, mark_radius=self.thickness, text_thickness=self.thickness, text_offset_x=self.x_offset, text_offset_y=self.y_offset)
                index += 1

        save_image_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_sfr_points.png'))
        cv2.imwrite(save_image_path, self.rgb)

    def _calcu_slope_intercept(self, point1, point2):
        delta = point2 - point1
        if delta[0] == 0:
            return 0, 0  # 避免除以零
        slope = delta[1] / delta[0]
        intercept = point1[1] - slope * point1[0]
        return slope, intercept

    def _preprocess_image(self, image):
        if self.mtf_debug or self.debug_flag:
            self.rgb = utils.raw_2_rgb(image)
        if self.max_radius > 0:
            mask = utils.generate_mask(image.shape, self.max_radius)
            image[mask == 0] = 0
        if self.bw_mode == BinaryMode.CANNY:
            image = cv2.blur(image, self.blur_size, 0)
            canny = cv2.Canny(image, self.canny_thresh[0], self.canny_thresh[0])
            kern = np.ones((7, 7), np.uint8)
            canny = cv2.dilate(canny, kern)
            if self.debug_flag:
                utils.show_image(image, self.visual_scalse_factor)
                utils.show_image(canny, self.visual_scalse_factor)
            contours, _ = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            bw_image = np.zeros_like(image, np.uint8)

            for contour in contours:
                area = cv2.contourArea(contour)  # 计算轮廓面积
                if area <= (image.size / 6):  # 只绘制满足面积范围的轮廓
                    print(area)
                    cv2.drawContours(bw_image, [contour], -1, 255, -1)
            kern = np.ones((15, 15), np.uint8)
            bw_image = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kern)
            if self.debug_flag:
                utils.show_image(bw_image, self.visual_scalse_factor)
                
        if self.bw_mode == BinaryMode.FIX_THRESH:
            norm_image = (image - image.min()) / (image.max() - image.min())
            _, bw_image = cv2.threshold(norm_image, self.bw_thresh, 255, cv2.THRESH_BINARY)
            if self.debug_flag:
                utils.show_image(image, self.visual_scalse_factor)
                # utils.show_image(bw_image, self.visual_scalse_factor)
            bw_image = bw_image.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.mask_erode_size)
            if self.apply_hull:
                conts, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_cont = max(conts, key = cv2.contourArea)  # key为指定函数， 面积函数
                # 使用凸包 获得图像边缘位置 找到掩膜
                hull = cv2.convexHull(max_cont)
                mask_tmp = np.zeros_like(image, np.uint8)
                mask_tmp = cv2.drawContours(mask_tmp, [hull], 0 , 255, -1)
                mask = cv2.erode(mask_tmp, kernel)  # 防止出现外围一圈白线
                # if self.debug_flag:
                #     utils.show_image(mask, self.visual_scalse_factor)
                bw_image[mask == 0] = 255
            bw_image = cv2.dilate(bw_image, kernel)
            if self.debug_flag:
                utils.show_image(~bw_image, self.visual_scalse_factor)
            bw_image = ~bw_image
        return bw_image

    def _select_half_block_dist(self, stats, centroid, center_xy, delta_dist, n_half_block):
        dist = utils.calcu_distance(centroid, center_xy)
        if self.debug_flag:
            print(f'half block distance: {np.sort(dist)}, \n thresh: {delta_dist}')
        index = np.where((dist > delta_dist[0]) & (dist < delta_dist[1]))[0]
        if len(index) != n_half_block:
            raise KeyError(f'locate_half_block: only find {len(index)} half, blocks, {n_half_block} needed!')
        return stats[index], centroid[index]

    def _get_block_coord(self, block, tl_x, tl_y, block_index, min_dist):
        center_edge = True   # 使用边的中心
        failFlag = True
        cur_inner_center = np.array([0, 0])
        block_size = block.shape
        # utils.show_image(block, self.visual_scalse_factor)
        _, bw_block = cv2.threshold(block, 150, 255, cv2.THRESH_OTSU)
        # utils.show_image(bw_block, self.visual_scalse_factor)
        kern = np.ones((9, 9), np.uint8)
        bw_block = cv2.morphologyEx(~bw_block, cv2.MORPH_OPEN, kern)
        # utils.show_image(bw_block, self.visual_scalse_factor)
        # 计算中心圆的质心
        inner_center_flag = False
        if np.any(np.array(self.inner_block_thresh) != 0):
            inner_center_flag = True
            _, centroid = self.find_connected_area(~bw_block, self.inner_block_thresh, 'Inner block')
            if len(centroid) > 0:
                dist = utils.calcu_distance(centroid, (block_size[1] // 2, block_size[0] // 2))
                center_index = np.argmin(dist)
                center_xy = centroid[center_index]
                cur_inner_center = center_xy
            else:
                cur_inner_center = np.array((block_size[1] / 2, block_size[0] / 2))


        contours, _ = cv2.findContours(bw_block, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # 计算轮廓的近似
            epsilon = 0.05 * cv2.arcLength(cnt, True)  # 单个轮廓, True表示轮廓封闭
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 检查是不是四边形
            if len(approx) == 4:
                blockXY = approx.reshape(4,2).copy()
                if (blockXY[:,0].max() - blockXY[:,0].min() > min_dist) and (blockXY[:,1].max() - blockXY[:,1].min() > min_dist):
                    failFlag = False
                break  # 如果找到一个四边形就停止

        if failFlag:  # 如果第一种方法fail  使用第二种方法
            pts = np.nonzero(bw_block)
            blockXY = np.column_stack(pts)[:, ::-1] 
        x_minus_y = blockXY[:, 0] - blockXY[:, 1]
        x_sum_y = blockXY.sum(axis=1)
        index_1 = np.argmax(x_minus_y)
        index_2 = np.argmin(x_sum_y)
        index_3 = np.argmin(x_minus_y)
        index_4 = np.argmax(x_sum_y)
        cur_block_corner_xy = np.zeros((4, 2))
        cur_block_corner_xy[0] = blockXY[index_1]
        cur_block_corner_xy[1] = blockXY[index_2]
        cur_block_corner_xy[2] = blockXY[index_3]
        cur_block_corner_xy[3] = blockXY[index_4]
        cur_roi_center_xy = np.zeros((4, 2))

        ## 是否抓取Edge中心的ROI
        if center_edge:
            cur_roi_center_xy = (cur_block_corner_xy + np.roll(cur_block_corner_xy, 1, axis=0)) / 2
            
        ## 抓取Edge与质心水平的ROI
        else:                
            ##  获取Edge的斜率

            for i in range(4):
                j = i - 1 if i - 1 >= 0 else 3
                k, b = self._calcu_slope_intercept(cur_block_corner_xy[i], cur_block_corner_xy[j])
                if i in [0, 2]:
                    cur_roi_center_xy[i] = np.array([(cur_inner_center[1] - b) / k if k != 0 else cur_block_corner_xy[i][0], cur_inner_center[1]])
                else:
                    cur_roi_center_xy[i] = np.array([cur_inner_center[0], k * cur_inner_center[0] + b])

        cur_inner_center += (tl_x, tl_y)
        cur_block_corner_xy += (tl_x, tl_y)
        cur_roi_center_xy += (tl_x, tl_y)
        cur_roi_center_xy = np.uint16(np.round(cur_roi_center_xy))
        if self.debug_flag:
            for prox in approx:
                prox += (tl_x, tl_y)
            cv2.drawContours(self.rgb, [approx], 0, (0, 255, 0), thickness=self.thickness)  # 用绿色标记四边形
            for point in approx:
                cv2.circle(self.rgb, tuple(point[0]),self.thickness, (0, 0, 255), -1)  # 用红色标记角点   
            if inner_center_flag:
                cv2.circle(self.rgb, tuple((cur_inner_center).astype(np.uint16)), self.thickness, (0, 0, 255), -1)  # 用红色标记角点   
                cv2.putText(self.rgb, str(block_index), tuple((cur_inner_center).astype(np.uint16)),cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 0, 255), self.thickness)
        return cur_roi_center_xy, cur_block_corner_xy, cur_inner_center

    def _debug_image(self, all_rect, value, save_path):
        if 4 in self.ny_freq:
            index = self.ny_freq.index(4)
        else:
            index = 0
        value = 100 * value
        
        for i, rect in enumerate(all_rect):
            text = f'R{i+1}:{value[i][index]:.1f}' if self.text_value else f'R_{i+1}'
            utils.draw_rect_with_text(self.rgb, rect[0], rect[1], rect[2], rect[3], text, text_size=self.text_size, text_thickness=self.thickness, rect_thickness=self.thickness, text_offset_x=self.x_offset, text_offset_y=self.y_offset)
        save_image_path = os.path.join(save_path, (utils.GlobalConfig.get_device_id() + '_sfr.png'))
        cv2.imwrite(save_image_path, self.rgb)

    @staticmethod
    def _group_and_sort_indices(points, y_threshold=50):
        """
        按照 y 坐标进行分组，并在每一行内部按照 x 坐标从小到大排序，返回点的索引
        :param points: 一个包含 (x, y) 坐标的列表
        :param y_threshold: 用于确定 y 坐标误差范围（默认50像素）
        :return: 按行分组并排序后的索引列表
        """
        # 获取带索引的点 (index, (x, y))
        indexed_points = list(enumerate(points))
        
        # 按 y 坐标排序
        indexed_points.sort(key=lambda p: p[1][1])  # 根据 y 坐标排序
        
        # 存储分组后的索引
        grouped_indices = []
        
        # 初始化第一行
        current_row = [indexed_points[0]]
        
        for i in range(1, len(indexed_points)):
            index, (x, y) = indexed_points[i]
            last_index, (_, last_y) = current_row[-1]  # 取当前行的最后一个点
            
            # 如果 y 坐标与上一点的 y 差值小于阈值，则视为同一行
            if abs(y - last_y) <= y_threshold:
                current_row.append((index, (x, y)))
            else:
                # 按 x 坐标排序后，提取索引并保存
                grouped_indices.append([idx for idx, _ in sorted(current_row, key=lambda p: p[1][0])])
                current_row = [(index, (x, y))]  # 开始新的行
        
        # 处理最后一行
        if current_row:
            grouped_indices.append([idx for idx, _ in sorted(current_row, key=lambda p: p[1][0])])
        
        return grouped_indices

    def find_connected_area(self, image, thresh, info=None, count=0):
        _, _, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        index = np.where((stats[1:, 4] > thresh[0]) & (stats[1:, 4] < thresh[1]))[0] + 1
        if self.debug_flag:
            print(f'{info} area: {np.sort(stats[1:, 4])}, \n thresh: {thresh}')
        if count != 0 and count != len(index):
            raise KeyError(f'locate_block: only find {len(index)} blocks, {count} needed!')
        
        valid_stats = stats[index]
        valid_centroid = centroids[index]
        return valid_stats, valid_centroid
    
    def localte_block_california(self, image, save_path):
        '''
        图像中心存在block， 所有block完整且不连接
        '''
        image_size = image.shape
        center_xy = (image_size[1] // 2, image_size[0] // 2)
        # 预处理
        bw_image = self._preprocess_image(image)
        
        # 获取bbox 和 质心
        block_stats, valid_centroid = self.find_connected_area(bw_image, self.block_thresh, 'Block', self.n_block)
        
        # 质心到圆心的距离
        dist_from_center = utils.calcu_distance(valid_centroid, center_xy)
        
        # block 按距离分组
        _, group_index = utils.group_data(dist_from_center, center_xy[0] // 15)
        
        # block 按角度分组
        group_sorted_index = []
        block_centroid = []
        for cur_group_index in group_index:
            cur_centroid = valid_centroid[cur_group_index]
            cur_sorted_index = utils.sort_order_index(cur_centroid, center_xy, self.delta_angle, self.clockwise)
            group_sorted_index.append(cur_group_index[cur_sorted_index])
            block_centroid.append(valid_centroid[cur_group_index[cur_sorted_index]])
            
        # 定位所有block相关坐标
        block_corner__xy, block_roi_center_xy, inner_block_center_xy = self._locate_all_block(image, block_stats, group_sorted_index, self.n_block, self.long_side)
        
        # ROI 中心示意图， 用于后续选择ROI
        if self.debug_flag:
            os.makedirs(save_path, exist_ok=True)
            self._visualize_all_points(block_roi_center_xy, save_path)
        return block_roi_center_xy, block_centroid, inner_block_center_xy, bw_image

    def localte_block_rgb(self, image, save_path):
        '''
        图像按照行列均匀排列
        '''
        image_size = image.shape
        # 预处理
        bw_image = self._preprocess_image(image)
        
        # 获取bbox 和 质心
        block_stats, valid_centroid = self.find_connected_area(bw_image, self.block_thresh, 'Block', self.n_block)
        
        group_index = SFR._group_and_sort_indices(valid_centroid, image_size[0] // 15)
        index = []
        for cur_group_index in group_index:
            index.extend(cur_group_index)
        block_centroid = valid_centroid[index]

        # 定位所有block相关坐标
        block_corner__xy, block_roi_center_xy, inner_block_center_xy = self._locate_all_block(image, block_stats, index, self.n_block, self.long_side)
        
        # ROI 中心示意图， 用于后续选择ROI
        if self.debug_flag:
            os.makedirs(save_path, exist_ok=True)
            self._visualize_all_points(block_roi_center_xy, save_path)
        return block_roi_center_xy, block_centroid, inner_block_center_xy, bw_image

    def locate_block_cv(self, image, save_path):
        '''
        图像中心存在block, block之间有连接
        '''
        image_size = image.shape
        center_xy = (image_size[1] // 2, image_size[0] // 2)
        # 预处理
        bw_image = self._preprocess_image(image)
        
        # 获取bbox 和 质心
        block_stats, valid_centroid = self.find_connected_area(bw_image, self.block_thresh, 'Block', self.n_block)
        
        # 质心到圆心的距离
        dist_from_center = utils.calcu_distance(valid_centroid, center_xy)
        
        # block 按距离分组
        _, group_index = utils.group_data(dist_from_center, center_xy[0] // 15)
        
        # block 按角度分组
        group_sorted_index = []
        block_centroid = []
        for cur_group_index in group_index:
            cur_centroid = valid_centroid[cur_group_index]
            cur_sorted_index = utils.sort_order_index(cur_centroid, center_xy, self.delta_angle, self.clockwise)
            group_sorted_index.append(cur_group_index[cur_sorted_index])
            block_centroid.append(valid_centroid[cur_group_index[cur_sorted_index]])
            
        # 定位所有block相关坐标
        block_corner_xy, block_roi_center_xy, inner_block_center_xy = self._locate_all_block(image, block_stats, group_sorted_index, self.n_block, self.long_side)

        # 计算half
        half_block_stats, half_centroid = self.find_connected_area(bw_image, self.half_block_thresh, 'Half Block')
        chart_center_xy = center_xy
        half_block_stats, half_centroid = self._select_half_block_dist(half_block_stats, half_centroid, chart_center_xy, self.half_bloack_dist_from_center, self.n_half_block)
        sorted_index = utils.sort_order_index(half_centroid, center_xy, self.delta_angle, self.clockwise)
        half_block_corner_xy, half_block_roi_center_xy, half_inner_block_center_xy = self._locate_all_block(image, half_block_stats, sorted_index, self.n_half_block, 0)
        block_roi_center_xy.extend(half_block_roi_center_xy)
        
        # ROI 中心示意图， 用于后续选择ROI
        if self.debug_flag:
            os.makedirs(save_path, exist_ok=True)
            self._visualize_all_points(block_roi_center_xy, save_path)
        return block_roi_center_xy, block_centroid, inner_block_center_xy, bw_image
    
    def get_roi_rect(self, bw_image, all_roi_center):
        test_size = 10
        n_roi = len(all_roi_center)
        all_rect = np.zeros((n_roi, 4), dtype=np.uint16)
        half_short = self.short_side // 2
        half_long = self.long_side // 2

        for i, cur_center in enumerate(all_roi_center):
            x_center, y_center = cur_center  # 解包中心点
            # 提取当前 ROI 区域
            cur_roi = bw_image[
                y_center - test_size : y_center + test_size,
                x_center - test_size : x_center + test_size]
            # 左半边-右半边
            diff_horizontal = np.abs(cur_roi[:, 0:test_size].mean() - cur_roi[:, test_size:].mean())
            # 上半边-下半边
            diff_vertical = np.abs(cur_roi[0:test_size, :].mean() - cur_roi[test_size:, :].mean())
            # 如果水平差值>垂直差值，说明是垂直方向的边，因此长边为行
            if diff_horizontal > diff_vertical :
                x_start = x_center - half_short
                y_start = y_center - half_long
                all_rect[i, :] = [x_start, y_start, self.short_side, self.long_side]
            # 如果水平差值<垂直差值，说明是水平方向的边，因此长边为列    
            else:
                x_start = x_center - half_long
                y_start = y_center - half_short
                all_rect[i, :] = [x_start, y_start, self.long_side, self.short_side]
        return all_rect

    def select_roi(self, blocks, roi_index):
        points_1d = np.concatenate(blocks)
        all_roi_center = points_1d[roi_index]
        return all_roi_center    
    
    def calcu_mtf(self, image, all_roi_rect, save_path):
        
        mtf_data = self.mtf.run(image, all_roi_rect)
        if self.csv_output or self.mtf_debug:
            os.makedirs(save_path, exist_ok=True)
        
        if self.csv_output:
            sperate_fre = True
            data = [utils.GlobalConfig.get_device_id(),]
            name = ['Device ID',]
            
            if sperate_fre and len(self.ny_freq) > 1:
                for fre_index in range(len(self.ny_freq)):
                    for i, mtf in enumerate(mtf_data):
                        name.append(f"ROI{i+1}_Ny_{self.ny_freq[fre_index]}")
                        data.append(str(100*mtf[fre_index]))   
            else:
                for i, mtf in enumerate(mtf_data):
                    name.extend([f"ROI{i+1}_Ny_{num}" for num in self.ny_freq])
                    data.extend([str(100*value) for value in mtf])        
            save_file_name = os.path.join(save_path, 'mtf_data.csv')
            utils.save_lists_to_csv(data, name, save_file_name)
        
        # 可视化
        if self.mtf_debug:
            self._debug_image(all_roi_rect, mtf_data, save_path)
        return mtf_data
    
#region    
# @time_it_avg(10)
def func(self, file_name, save_path):
    
    image = utils.load_image(file_name, self.image_tpye, self.image_size, self.crop_tblr)
    if self.sub_black_level:
        image = utils.sub_black_level(image, self.black_level)
    
    if self.bayer_pattern != 'Y':
        image = utils.bayer_2_y(image, self.bayer_pattern)
    
    if image.dtype == np.uint16:
        image = (image >> 2).astype(np.uint8)
    
    # 定位block
    block_roi_center_xy, block_centroid, inner_block_center_xy, points_xy = self.localte_block_rgb(image, save_path)
    # block_roi_center_xy, block_centroid, inner_block_center_xy, points_xy = self.locate_block_cv(image, save_path)
    
    # 选择roi
    all_roi_center_xy = self.select_roi(block_roi_center_xy, self.roi_index)
    
    # 创建rect
    all_roi_rect = self.get_roi_rect(image, all_roi_center_xy)
        
    # 计算mtf
    self.calcu_mtf(image, all_roi_rect)
        
        
if __name__ == '__main__':
    file_name = r'D:\Code\CameraTest\image\ET\sfr.raw'
    save_path = r'D:\Code\CameraTest\result'
    config_path = r'D:\Code\CameraTest\Config\config_et.yaml'
    sfr = SFR(config_path, 1)
    utils.process_files(file_name, sfr.func, '.raw', save_path)
    print('sfr finished!') 
#endregion