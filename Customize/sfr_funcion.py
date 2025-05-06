from Common import utils
import numpy as np

def select_point_california(centroid, center_xy, n_point, point_dist_from_center, clockwise, local_debug=False):
    '''
    找图中与圆心等距的point，并逆时针排序
    1 -------- 2
    |          |
    |          |
    4 -------- 3 
    '''
    dist_2_center = utils.calcu_distance(centroid, center_xy)
    if local_debug:
        print(f'point distance to center:{np.sort(dist_2_center)}')
    index = np.where((dist_2_center > point_dist_from_center[0]) & (dist_2_center < point_dist_from_center[1]))[0]
    if len(index) != n_point:
        raise KeyError(f'point dist to center: {dist_2_center} /n {len(index)} found, but {n_point} needed!')
    point_centroid = centroid[index]
    sorted_index = utils.sort_order_index(point_centroid, center_xy, 15, clockwise)
    sorted_point = point_centroid[sorted_index]
    return sorted_point

def select_point_cv(centroid, center_xy, n_point, point_dist_from_center, clockwise, local_debug=False):
    '''
    找图中与圆心等距的point，并逆时针排序
    1 -------- 2
    |          |
    |    0     |
    |          |
    4 -------- 3 
    '''
    dist_2_center = utils.calcu_distance(centroid, center_xy)
    if local_debug:
        print(f'point distance to center:{np.sort(dist_2_center)}')
    index = np.where((dist_2_center > point_dist_from_center[0]) & (dist_2_center < point_dist_from_center[1]))[0]
    if len(index) != n_point-1:
        raise KeyError(f'point dist to center: {dist_2_center} /n {len(index)} found, but {n_point} needed!')
    point_centroid = centroid[index]
    sorted_index = utils.sort_order_index(point_centroid, center_xy, 15)
    sorted_point = [center_xy,]
    sorted_point.extend(point_centroid[sorted_index]) 
    return sorted_point

def select_point_et(image, center_xy, thresh, points_offset_xy, points_roi_size, n_point, point_dist_from_center, clockwise, debug_flag=False):
    '''
    1     2
    
    3     4
    
    '''
    center_xy = [round(center_xy[0]), round(center_xy[1])]
    tl_center = [center_xy[0] - points_offset_xy[0], center_xy[1] - points_offset_xy[1]]
    tr_center = [center_xy[0] + points_offset_xy[0], center_xy[1] - points_offset_xy[1]]
    bl_center = [center_xy[0] - points_offset_xy[0], center_xy[1] + points_offset_xy[1]]
    br_center = [center_xy[0] + points_offset_xy[0], center_xy[1] + points_offset_xy[1]]
    center_roi_xy = [points_roi_size[0] // 2, points_roi_size[1] // 2]
    def calcu_coord_delta(cur_center, roi_size, thresh, center_roi_xy):
        row_start, row_end, col_start, col_end = utils.get_rect(cur_center, roi_size)
        _, centroid = utils.find_connected_area(image[row_start: row_end, col_start: col_end], thresh, info='points', debug_flag=False)
        if len(centroid) == 0:
            raise TypeError('ERROR', f'select point:  cant find points thresh:{thresh}')
        if len(centroid) > 1:
            dists = utils.calcu_distance(centroid, center_roi_xy)
            index = np.argmin(dists)
            center = centroid[index]
        else:
            center = centroid[0]
        delta_xy = [center[0] - center_roi_xy[0], center[1] - center_roi_xy[1]]
        real_xy = np.array(cur_center) + np.array(delta_xy)
        return real_xy
    
    tl = calcu_coord_delta(tl_center, points_roi_size, thresh, center_roi_xy)
    tr = calcu_coord_delta(tr_center, points_roi_size, thresh, center_roi_xy)
    ll = calcu_coord_delta(bl_center, points_roi_size, thresh, center_roi_xy)
    lr = calcu_coord_delta(br_center, points_roi_size, thresh, center_roi_xy)
    points_xy = np.array([tl,tr,ll,lr])
    return points_xy
     
def rotation_rgb(point_xy_0, point_xy_1, point_xy_2, point_xy_3, point_xy_4):
    '''
    #      2
    #  3   0   1 
    #      4
    Rotation_1 = atan((Y0-Y1)/(X1-X0)) 
    Rotation_2 = atan((X2-X0)/(Y2-Y0)) 
    Rotation_3 = atan((Y0-Y3)/(X3-X0)) 
    Rotation_4 = atan((X4-X0)/(Y4-Y0))
    '''
    rotation_1 = np.arctan((point_xy_0[1] - point_xy_1[1]) / (point_xy_1[0] - point_xy_0[0])) 
    rotation_2 = np.arctan((point_xy_2[0] - point_xy_0[0]) / (point_xy_2[1] - point_xy_0[1])) 
    rotation_3 = np.arctan((point_xy_0[1] - point_xy_3[1]) / (point_xy_0[0] - point_xy_3[0])) 
    rotation_4 = np.arctan((point_xy_4[0] - point_xy_0[0]) / (point_xy_4[1] - point_xy_0[1])) 
    rotation = (rotation_1 + rotation_2 + rotation_3 + rotation_4) * 0.25
    return rotation, rotation_1, rotation_2, rotation_3, rotation_4

def rotation_cv(point_a, point_b, point_c, point_d, point_e):
    '''
    #  b     a
    #     e 
    #  c     d
    Rotation_a = 45 - |arctan((Ay-Ey)/(Ax-Ex))|
    Rotation_b = |arctan((By-Ey)/(Bx-Ex))|-45
    Rotation_c = 45 - |arctan((Cy-Ey)/(Cx-Ex))|
    Rotation_d = |arctan((Dy-Ey)/(Dx-Ex))| -45
    '''
    
    rotation_a = np.radians(45) - np.abs(np.arctan((point_a[1] - point_e[1]) / (point_a[0] - point_e[0])))
    rotation_b = np.abs(np.arctan((point_b[1] - point_e[1]) / (point_b[0] - point_e[0]))) - np.radians(45)
    rotation_c = np.radians(45) - np.abs(np.arctan((point_c[1] - point_e[1]) / (point_c[0] - point_e[0])))
    rotation_d = np.abs(np.arctan((point_d[1] - point_e[1]) / (point_d[0] - point_e[0]))) - np.radians(45)
    rotation = (rotation_a + rotation_b + rotation_c + rotation_d) * 0.25
    return rotation, rotation_a, rotation_b, rotation_c, rotation_d

def rotation_et(points_xy):
    '''
    1     2

    3     4
    也就是
    5      6
    
    17     18
    '''
    point5 = points_xy[0]
    point6 = points_xy[1]
    point17 = points_xy[2]
    point18 = points_xy[3]
    
    Rotation_TL_BR = np.arctan((point5[1] - point18[1])/(point5[0] - point18[0])) - np.arctan(1)
    Rotation_TR_BL = np.arctan(1) - np.abs(np.arctan((point6[1] - point17[1])/(point6[0] - point17[0])))
    Rotation_Mean = (Rotation_TL_BR + Rotation_TR_BL) * 0.5
    return Rotation_TL_BR, Rotation_TR_BL, Rotation_Mean

def fov_rgb(point_xy_1, point_xy_2, point_xy_3, point_xy_4, image_size, length_h_real_mm, length_v_real_mm, chart_distance):
    length_h_12 = utils.calcu_distance(point_xy_1, point_xy_2)
    length_h_34 = utils.calcu_distance(point_xy_3, point_xy_4)
    length_v_13 = utils.calcu_distance(point_xy_1, point_xy_3)
    length_v_24 = utils.calcu_distance(point_xy_2, point_xy_4)
    length_h_mean = (length_h_12 + length_h_34) * 0.5
    length_v_mean = (length_v_13 + length_v_24) * 0.5
    FOV_factor_h = image_size[1] / length_h_mean
    FOV_factor_v = image_size[0] / length_v_mean
    length_FOV_h_real_mm = length_h_real_mm * FOV_factor_h 
    length_FOV_v_real_mm = length_v_real_mm * FOV_factor_v
    length_FOV_d_real_mm = np.sqrt(length_FOV_h_real_mm ** 2, length_FOV_v_real_mm ** 2) 
    FOV_H = 2 * np.arctan(length_FOV_h_real_mm/2/chart_distance) 
    FOV_V = 2 * np.arctan(length_FOV_v_real_mm/2/chart_distance) 
    FOV_D = 2 * np.arctan(length_FOV_d_real_mm/2/chart_distance) 
    return FOV_H, FOV_V, FOV_D

def fov_Cv(point_a, point_b, point_c, point_d, point_g, point_h, point_j, point_k, 
           image_circle, distD_design_percent, distV_design_percent, distH_design_percent, fov_design):
    dist_AC = utils.calcu_distance(point_a, point_c)[0]
    dist_BD = utils.calcu_distance(point_b, point_d)[0]
    dist_GJ = utils.calcu_distance(point_g, point_j)[0]
    dist_HK = utils.calcu_distance(point_h, point_k)[0]
    dist_D = (dist_AC + dist_BD) * 0.5
    dist_D_image_percent = dist_D / image_circle
    dist_V_image_percent = dist_GJ / image_circle
    dist_H_image_percent = dist_HK / image_circle
    fov_d = fov_design * (distD_design_percent / dist_D_image_percent)
    fov_v = fov_design * (distV_design_percent / dist_V_image_percent)
    fov_h = fov_design * (distH_design_percent / dist_H_image_percent)
    return fov_d, fov_v, fov_h
 
def fov_et(points_xy, image_circle, fov_design, fov_ratio):
    '''
    1    2

    3    4
    也就是
    5      6

    17     18
    '''
    dist_5_18 = utils.calcu_distance(points_xy[0], points_xy[3])
    dist_6_17 = utils.calcu_distance(points_xy[1], points_xy[2])
    fov_d = ((4 * image_circle) / (dist_5_18 + dist_6_17)) * fov_design * fov_ratio
    return fov_d[0]
    
def tilt_rgb(point_xy_1, point_xy_2, point_xy_3, point_xy_4):
    '''
    tilt_X = atan((X1-X3)/(Y3-Y1)) + atan((X4-X2)/(Y4-Y2))
    tilt_Y = atan((Y1-Y2)/(X2-X1)) + atan((Y4-Y3)/(X4-X3)) 
    '''
    tilt_x = np.arctan((point_xy_1[0] - point_xy_3[0]) / (point_xy_3[1] - point_xy_1[1])) + np.arctan((point_xy_4[0] - point_xy_2[0]) / (point_xy_4[1] - point_xy_2[1])) 
    tilt_y = np.arctan((point_xy_1[1] - point_xy_2[1]) / (point_xy_2[0] - point_xy_1[0])) + np.arctan((point_xy_4[1] - point_xy_3[1]) / (point_xy_4[0] - point_xy_3[0])) 
    return tilt_x, tilt_y
    
def tilt_cv(offset_x, offset_y, pixel_size, efl):
    '''
    Pan = atan((oc_x * pixel_size) / efl) 
    Tile = atan((oc_y * pixel_size) / efl) 
    '''
    pan = np.arctan((offset_x * pixel_size) / efl)
    tilt = np.arctan((offset_y * pixel_size) / efl) 
    return pan, tilt

def pointing_oc_et(points_xy, center_xy):
    avg_xy = points_xy.mean(axis=0)
    oc_x, oc_y = avg_xy + 0.5
    offset_x = oc_x - center_xy[0]
    offset_y = oc_y - center_xy[1]
    oc_r = np.sqrt(offset_x**2 + offset_y**2)
    return oc_x, oc_y, offset_x, offset_y, oc_r

def brightness_cv(image, roi_centers, roi_size):
    #  B     A
    #
    #  C     D
    brightness = []
    for roi_center in roi_centers:
        row_start, row_end, col_start, col_end = utils.get_rect(roi_center, roi_size)
        brightness.append(image[row_start: row_end, col_start: col_end].mean())
    return brightness

def distortion_1():
    pass
