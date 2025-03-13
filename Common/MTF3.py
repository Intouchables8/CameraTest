import numpy as np
from .MTF3_CSupport import sfrmat3
from .utils import *
# from .MTF3_Support import sfrmat3, calcny_freqMtf
class MTF3:
    def __init__(self, ny_freq, gamma=1, interval_unit=1, weight=(0.2126,0.7152,0.0722),oecfname='none'):
        allowed = {2, 4, 8}
        mapping = {2: 0.25, 4: 0.125, 8: 0.0625}
        for num in ny_freq:
            if num not in allowed:
                raise ValueError(f"MTF3: only allow ny frequnce 2 4 8")
    
        # 映射列表中的数字
        mapped_list = [mapping[num] for num in ny_freq]
        self.ny_freq = mapped_list
        self.n_ny = len(ny_freq)
        self.SFRclass = sfrmat3(gamma, interval_unit, weight, oecfname)
        # 执行相关代码
    def _calcu_mtf(self, roi):
        mtf_peak_percent = (0.5,)
        _mtfData, _mtf_peak_percentFreq, _ny_freqMTF, _fitCoeff = self.SFRclass.calcOneRoiStart_API(roi.copy(), mtf_peak_percent, self.ny_freq)
        return _ny_freqMTF
    
    def run(self, image, all_roi_rect):
        n_roi = len(all_roi_rect)
        mtf_data = np.zeros((n_roi, self.n_ny) ,np.float32)
        for i, rect in enumerate(all_roi_rect):   # rect x y w h
            roi = image[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]  
            mtf_data[i, :] = self._calcu_mtf(roi)
        return mtf_data
        






