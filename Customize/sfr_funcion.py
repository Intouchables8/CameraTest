import numpy as np
from Common import utils


def select_point_california(centroid, center_xy, n_point, point_dist_from_center, local_debug=False):
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
    sorted_index = utils.sort_order_index(point_centroid, center_xy, 15)
    sorted_point = point_centroid[sorted_index]
    return sorted_point

def select_point_cv(centroid, center_xy, n_point, point_dist_from_center, local_debug=False):
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
    


def rotation_1():
    pass

def pointing_oc_1():
    pass

def tilt_1():
    pass

def distortion_1():
    pass

def brightness_1():
    pass





# def FOV(self):
#     rad = 180 / np.pi
#     point69 = innr_block_center_xy[13]
#     point70 = innr_block_center_xy[14]
#     point68 = innr_block_center_xy[16]
#     point67 = innr_block_center_xy[15]
#     dis6769 = np.sqrt((point67[0] - point69[0])**2 + (point67[1] - point69[1])**2)
#     dis6870 = np.sqrt((point68[0] - point70[0])**2 + (point68[1] - point70[1])**2)
#     DFOV = ((4 * imageCircle) / ((dis6769 + dis6870) * pixelSize)) * FOV_design * 0.75 

#     fovData = {
#         f'SFR_{chartDistance}_D_FOV': str(DFOV),
#     }

# def Rotation(self):
#     rad = 180 / np.pi
#     height, width = image_size
#     point69 = innr_block_center_xy[13]
#     point0 = OC_Center
#     point70 = innr_block_center_xy[14]
#     point68 = innr_block_center_xy[16]
#     point67 = innr_block_center_xy[15]
#     HWrad = np.arctan(height/width)
#     rotationA = (np.arctan((point69[1] - point0[1]) / (point69[0] - point0[0])) - HWrad) * rad
#     rotationB = (HWrad - abs(np.arctan((point70[1] - point0[1]) / (point70[0] - point0[0])))) * rad
#     rotationC = (HWrad - abs(np.arctan((point68[1] - point0[1]) / (point68[0] - point0[0])))) * rad
#     rotationD = (np.arctan((point67[1] - point0[1]) / (point67[0] - point0[0])) - HWrad) * rad
#     rotation = np.array([rotationA, rotationB, rotationC, rotationD])
#     rotationMean = np.mean(rotation)
#     rotationMedian = np.median(rotation)
#     rotationStd = np.std(rotation, ddof=1)
#     rotationData = {
#         f'SFR_{chartDistance}_Rotation_Mean': str(rotationMean),
#         f'SFR_{chartDistance}_Rotation_Median': str(rotationMedian),
#         f'SFR_{chartDistance}_Rotation_Std': str(rotationStd),                 
#     }
    
# def PointOC(self):
#     imgH, imgW = image_size
#     ocX = np.mean((innr_block_center_xy[13][0], innr_block_center_xy[14][0], innr_block_center_xy[15][0], innr_block_center_xy[16][0])) 
#     ocY = np.mean((innr_block_center_xy[13][1], innr_block_center_xy[14][1], innr_block_center_xy[15][1], innr_block_center_xy[16][1])) 
#     offsetX = ocX - imgW / 2 + 0.5
#     offsetY = ocY - imgH / 2 + 0.5
#     offsetX = offsetX
#     offsetY = offsetY
#     offsetR = np.sqrt(offsetX**2 + offsetY**2)
#     pointOcData = {
#                     f'SFR_{chartDistance}_Pointing_OC_X':str(offsetX),
#                     f'SFR_{chartDistance}_Pointing_OC_Y':str(offsetY)
#     }

# def Tilt(self):
#     rad = 180 / np.pi
#     tilt_X = np.arctan((offsetY * pixelSize) / EFL) * rad
#     tilt_Y = np.arctan((offsetX * pixelSize) / EFL) * rad

#     tiltData = {
#         f'SFR_{chartDistance}_Tilt_X':str(tilt_X),
#         f'SFR_{chartDistance}_Tilt_Y':str(tilt_Y)
#     }

# def Distortion(self):
#     dis7879 = np.sqrt((point78[0] - point79[0])**2 + (point78[1] - point79[1])**2)
#     dis7780 = np.sqrt((point77[0] - point80[0])**2 + (point77[1] - point80[1])**2)
#     A = (dis7879 + dis7780) / 2
#     B = np.sqrt((point10[0] - point12[0])**2 + (point10[1] - point12[1])**2)
#     distortion = (A-B) / A

#     point69 = innr_block_center_xy[13]
#     point0 = OC_Center
#     point70 = innr_block_center_xy[14]
#     point68 = innr_block_center_xy[16]
#     point67 = innr_block_center_xy[15]


#     distortionData = {
#                             f'SFR_{chartDistance}_Distortion': str(distortion * 100),
#                             f'SFR_{chartDistance}_Point0_X': str(point0[0]),
#                             f'SFR_{chartDistance}_Point0_Y': str(point0[1]),

#                             f'SFR_{chartDistance}_Point67_X': str(point67[0]),
#                             f'SFR_{chartDistance}_Point67_Y': str(point67[1]),

#                             f'SFR_{chartDistance}_Point68_X': str(point68[0]),
#                             f'SFR_{chartDistance}_Point68_Y': str(point68[1]),

#                             f'SFR_{chartDistance}_Point69_X': str(point69[0]),
#                             f'SFR_{chartDistance}_Point69_Y': str(point69[1]),

#                             f'SFR_{chartDistance}_Point70_X': str(point70[0]),
#                             f'SFR_{chartDistance}_Point70_Y': str(point70[1]),

#                             f'SFR_{chartDistance}_Point77_X': str(point77[0]),
#                             f'SFR_{chartDistance}_Point77_Y': str(point77[1]),

#                             f'SFR_{chartDistance}_Point78_X': str(point78[0]),
#                             f'SFR_{chartDistance}_Point78_Y': str(point78[1]),

#                             f'SFR_{chartDistance}_Point79_X': str(point79[0]),
#                             f'SFR_{chartDistance}_Point79_Y': str(point79[1]),

#                             f'SFR_{chartDistance}_Point80_X': str(point80[0]),
#                             f'SFR_{chartDistance}_Point80_Y': str(point80[1]),

#     }
#     return

# def roiBrightness(self, YImahge, _len02F, roiSize):
#     deltaX = _len02F * np.cos(np.radians(45))
#     deltaY = _len02F * np.sin(np.radians(45))
#     center = (round(OC_Center[0] + deltaX), round(OC_Center[1] - deltaY)) 
#     centerRoi = YImahge[center[1] - roiSize[1] // 2 : center[1] + roiSize[1] // 2 , center[0] - roiSize[0] // 2 : center[0] + roiSize[0] // 2]
#     cv2.rectangle(rgb, (center[0] - roiSize[1] // 2, center[1] - roiSize[1] // 2), (center[0] + roiSize[0] // 2, center[1] + roiSize[1] // 2), (0,255,255), thickness=3) 
#     roiBrightness = centerRoi.mean()
#     brightnessData = {
#         f'SFR_{chartDistance}_Brightness': str(roiBrightness)
#     }
#     return 


    
# def debugImage(self):
#     # block顶点坐标与中心坐标
#     # I_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    
#     for block in block_corner__xy:
#         for cor in block:   
#             cv2.circle(rgb, (int(cor[0]), int(cor[1])), 4, (255, 165, 0), -1)    
#     # for block in block_roi_center_xy:
#     #     for center in block:   
#     #         cv2.circle(rgb, (int(center[0]), int(center[1])), 3, (0, 255, 0), -1)  
#     #         text = 'ROI:' + str(ROI) 
#     #         cv2.putText(rgb, text,  (int(center[0]+3), int(center[1]+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
#     #         ROI = ROI + 1
#     ROI = 1
#     for coor in AllROI:
#         cv2.rectangle(rgb, (coor[0], coor[1]), (coor[2], coor[3]), (0,255,255), thickness=3)     
#         text = 'ROI:' + str(ROI) 
#         cv2.putText(rgb, text,  (int(coor[0]+3), int(coor[1]+3)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
#         ROI += 1

#     imageFileName = savePath / (deviceId +'_SFR_ROI.png')
#     cv2.imwrite(imageFileName, rgb)                   
#     return True



