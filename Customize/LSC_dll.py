import ctypes
import sys
import os
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(str(ROOTPATH))
from Common import utils

class CalStructGeneral(ctypes.Structure):
    _fields_ = [
        ("R_LSC", ctypes.POINTER(ctypes.c_ushort)),
        ("Gr_LSC", ctypes.POINTER(ctypes.c_ushort)),
        ("Gb_LSC", ctypes.POINTER(ctypes.c_ushort)),
        ("B_LSC", ctypes.POINTER(ctypes.c_ushort)),
        ("AWB", ctypes.POINTER(ctypes.c_ushort)),
        ("AWB_ave", ctypes.POINTER(ctypes.c_float)),
        ("r_max", ctypes.c_float),
        ("gr_max", ctypes.c_float),
        ("gb_max", ctypes.c_float),
        ("b_max", ctypes.c_float),
        ("R_LSC_OG", ctypes.POINTER(ctypes.c_float)),
        ("Gr_LSC_OG", ctypes.POINTER(ctypes.c_float)),
        ("Gb_LSC_OG", ctypes.POINTER(ctypes.c_float)),
        ("B_LSC_OG", ctypes.POINTER(ctypes.c_float))
    ]
dllPath = r'.\\Common\\Qualcomm_LSC_U_64.dll'
dll = ctypes.WinDLL(dllPath)
# 定义函数的参数类型
dll.LensCorrectionLibRaw10_R.argtypes = [
                ctypes.POINTER(ctypes.c_ushort),    # Bayer数据指针
                ctypes.c_int,                       # 宽度 4032
                ctypes.c_int,                       # 高度 3024
                ctypes.c_int,                       # Bayer Pattern 0
                ctypes.c_short,                     # R通道blacklevel 0
                ctypes.c_short,                     # GR通道blacklevel 0
                ctypes.c_short,                     # GB通道blacklevel 0
                ctypes.c_short,                     # B通道blacklevel 0
                ctypes.c_bool,                      # false 
                ctypes.POINTER(CalStructGeneral),   # 结果集指针 (这个会根据结构体实际情况定义)
                ctypes.c_int,                       # sizeFactor 8
                ctypes.c_bool                       # 按图像中心取值 true
                ]
dll.LensCorrectionLibRaw10_R.restype = ctypes.c_int  # 返回值类型

def calcu_lsc(image, table_size, bayer_pattern, size_factor=8):
    '''
    #define RGGB_PATTERN	0
    #define GRBG_PATTERN	1
    #define BGGR_PATTERN	2
    #define GBRG_PATTERN	3
    '''
    if bayer_pattern == 'RGGB':
        mode = 0
    elif bayer_pattern == 'GRBG':
        mode = 1
    elif bayer_pattern == 'BGGR':
        mode = 2
    elif bayer_pattern == 'GBRG':
        mode = 3
        
    height, width = image.shape
    raw_data = image.flatten()
    bayer_ptr = raw_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    # 创建CalStructGeneral实例
    cal_result = CalStructGeneral()

    # 需要为结构体内的数组分配内存
    len_result = table_size[0] * table_size[1]
    cal_result.R_LSC = (ctypes.c_ushort * len_result)()
    cal_result.Gr_LSC = (ctypes.c_ushort * len_result)()
    cal_result.Gb_LSC = (ctypes.c_ushort * len_result)()
    cal_result.B_LSC = (ctypes.c_ushort * len_result)()
    cal_result.AWB = (ctypes.c_ushort * 3)()
    cal_result.AWB_ave = (ctypes.c_float * 4)()
    cal_result.R_LSC_OG = (ctypes.c_float * len_result)()
    cal_result.Gr_LSC_OG = (ctypes.c_float * len_result)()
    cal_result.Gb_LSC_OG = (ctypes.c_float * len_result)()
    cal_result.B_LSC_OG = (ctypes.c_float * len_result)()

    cal_result_ptr = ctypes.cast(ctypes.pointer(cal_result), ctypes.c_void_p)

    try:
        ret = dll.LensCorrectionLibRaw10_R(
                bayer_ptr,              # Bayer数据
                width,                  # 宽度
                height,                 # 高度
                mode,                   # Bayer Pattern (假设为0)
                0, 0, 0, 0,             # 各通道的black level
                False,                  # 是否为9*7
                cal_result,             # 使用 c_void_p 类型传递结构体指针
                size_factor,            # WB取值区域
                True                    # 是否使用中心
                )
        if ret == 0:
            result = {}
            for i in range(len_result):
                result[f'LSC_R_{i}'] = cal_result.R_LSC[i]
                result[f'LSC_Gr_{i}'] =cal_result.Gr_LSC[i]
                result[f'LSC_Gb_{i}'] =cal_result.Gb_LSC[i]
                result[f'LSC_B_{i}'] =cal_result.B_LSC[i]
            result['R_Gr'] = cal_result.AWB[0]
            result['B_Gr'] = cal_result.AWB[1]
            result['Gb_Gr'] = cal_result.AWB[2]
            return result   
        else:
            print(f"Error during calibration. Return code: {ret}")

    except Exception as e:
        print(f"调用 LensCorrectionLibRaw10_R 时出错: {e}")

if __name__ == '__main__':
    file_name =r'E:\Wrok\ERS\Oregon\对标数据\light\CU\light field image.raw'
    image_size = (3024, 4032) # row col
    table_size = (17, 13)
    black_level = 64
    bayer_pattern = 'RGGB'
    image = utils.load_image(file_name, "RAW10", image_size)
    image = utils.sub_black_level(image, black_level)
    result = calcu_lsc(image, table_size, bayer_pattern)
    pass
    





