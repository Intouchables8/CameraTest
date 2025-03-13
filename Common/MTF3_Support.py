import numpy as np
import cv2

eps = 1e-14

# cv2.fitLine(points,cv2.DIST_L2,0,0.001,0.001)
# 拟合直线（最小距离方法）
def LineFit(Pts):
    L = np.zeros((2,2),float)
    Xavg = 0
    Yavg = 0
    for Pt in Pts:
        Xavg += Pt[0]
        Yavg += Pt[1]
    Xavg /= Pts.shape[0]
    Yavg /= Pts.shape[0]
    for Pt in Pts:
        L[0,0] += Pt[0]*(Pt[0]-Xavg)
        L[0,1] += Pt[1]*(Pt[0]-Xavg)
        L[1,0] += Pt[0]*(Pt[1]-Yavg)
        L[1,1] += Pt[1]*(Pt[1]-Yavg)
    E_Value = np.linalg.eig(L)
    tData = E_Value[0][0]
    Res = E_Value[1][:,0]
    for i in range(E_Value[0].shape[0]):
        if np.iscomplex(E_Value[0][i]):
            continue
        if tData > E_Value[0][i]:
            tData = E_Value[0][i]
            Res = E_Value[1][:,i]
    return (Res[0],Res[1],-Res[0]*Xavg-Res[1]*Yavg)

def gamma_trans(src, gamma=2.0):
    # #具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    # gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    # gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # #实现映射用的是Opencv的查表函数
    # dst = cv2.LUT(src,gamma_table)
    scale = np.max(src).astype(np.float32)
    dst = ((src.astype(np.float32) / scale) ** gamma) * scale
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst

def Line2Point(lineparam,roisize):
    # lineparam-> (vx,vy,x0,y0)
    # (x-x0) / vx = (y-y0) / vy
    #(y-y0) / vy * vx + x0
    x1 = (0-lineparam[3]) / lineparam[1] * lineparam[0] + lineparam[2]# y=0
    x2 = (roisize[1]-lineparam[3]) / lineparam[1] * lineparam[0] + lineparam[2]# y=max
    y1 = (0-lineparam[2]) / lineparam[0] * lineparam[1] + lineparam[3]
    y2 = (roisize[0]-lineparam[2]) / lineparam[0] * lineparam[1] + lineparam[3]
    # x=0 y=0 xy
    pts = []
    if(x1 > 0 and x1 < roisize[0]):
        pts.append((float(x1),0))
    if(x2 > 0 and x2 < roisize[0]):
        pts.append((float(x2),roisize[1]))
    if(y1 > 0 and y1 < roisize[0]):
        pts.append((0,float(y1)))
    if(y2 > 0 and y2 < roisize[0]):
        pts.append((roisize[0],float(y2)))
    return tuple(pts)

def clipping(a, low, high, thresh1):
    # function [nlow, nhigh, status] = clipping(a, low, high, thresh1)
    # [n, status] = clipping(a)
    # Function checks for clipping of data array
    # a = array
    # low = low clip value
    # high = high clip value
    # thresh1 = threshhold fraction used for warning, if =0 all
    #           clipping is reported
    status = 1
    ts = a.shape

    n = ts[0]*ts[1]
    if len(ts) == 1:
        nhigh = np.zeros((1, 1))
        nlow =  np.zeros((1, 1))
        nlow[0] = np.sum(a <= low)
        nhigh[0] = np.sum(a >= high)
    else:
        nhigh = np.zeros((ts[2], 1))
        nlow =  np.zeros((ts[2], 1))
        for i in range(ts[2]):
            nlow = np.sum(a <= low)
            nhigh = np.sum(a >= high)
    nhigh = nhigh / n
    if np.sum(nlow) > thresh1:
        print([' *** Warning: low clipping in record ', str(nlow)])
        status = 0
    if np.sum(nhigh) > thresh1:
        print([' *** Warning: high clipping in record ', str(nhigh)])
        status = 0
    nlow = nlow / n
    return nlow, nhigh, status

def deriv1(a, nlin, npix, fil):
    # function  [b] = deriv1(a, nlin, npix, fil)
    # [b] = deriv1(a, nlin, npix, fil)
    #  Computes first derivative via FIR (1xn) filter
    #  Edge effects are suppressed and vector size is preserved
    #  Filter is applied in the npix direction only
    #   a = (nlin, npix) data array
    #   fil = array of filter coefficients, eg [-0.5 0.5]
    #   b = output (nlin, npix) data array
    #  Author: Peter Burns, 1 Oct. 2008
    #  Copyright (c) 2007 Peter D. Burns

    b = np.zeros((nlin, npix))
    nk = len(fil)
    for i in range(0,nlin):
        #ignore edge effects, preserve size
        b[i, nk-1:npix] = np.convolve(a[i,:],fil,'valid')
        b[i, nk-2] = b[i, nk-1]
    return b

def centroid(x):
    # function [loc] = centroid(x)
    #  Returns centroid location of a vector
    #   x = vector
    #   loc = centroid in units of array index
    #       = 0 error condition avoids division by zero, often due to clipping
    #  Author: Peter Burns, 1 Oct. 2008
    #  Copyright (c) 2007 Peter D. Burns
    n = max([x.shape[0],x.shape[1]])
    nidx = np.arange(1,n+1,1)
    sumx = x.sum()
    if sumx < 1e-4:
        loc = 0
    else:
        loc = np.sum(nidx * x) / sumx
    return loc

def cent(a, center):
    # Usage: [b] = cent(a, center)
    # Matlab function cent, shift of one-dimensional
    # array, so that a(center) is located at b(round((n+1)/2).
    # Written to shift a line-spread function array prior to 
    # applying a smoothing window.
    #  a      = input array
    #  center = location of signal center to be shifted
    #  b      = output shifted array
    #
    #  Author: Peter Burns, 1 Oct. 2008
    #  Copyright (c) 2007 Peter D. Burns
    ts = a.shape
    n = max([ts[0],ts[1]])
    b = np.zeros( ts,np.float32)
    mid = int((n+1)/2 + 0.5)
    del0 = center - mid
    
    if del0 > 0:
        nindex = np.array(range(0,n-del0),np.int32) + del0
        b[0,nindex-del0] = a[0,nindex]
    elif del0 < 1: #应该是del<0
        nindex = np.array(range(-del0,n),np.int32) + del0
        b[0,nindex-del0] = a[0,nindex]
    else :
        b = a
    return b



def apply_hamming_window(n,mid):
    # function generates a general asymmetric Hamming-type window array.
    # If mid = (n+1)/2 then the usual symmetric Hamming window is returned
    #  n = length of array
    #  mid = midpoint (maximum) of window function
    #  data = window array (nx1)
    data = np.zeros((1,n))
    wid = max(mid-1, n-mid)
    mid = mid-1
    for i in range(0,n):
        data[0,i] =  0.54 + 0.46 * np.cos( np.pi * (i-mid) / wid )
    return data

def fir2fix(n, m):
    # Correction for MTF of derivative (difference) filter
    #  n = frequency data length [0-half-sampling (Nyquist) frequency]
    #  m = length of difference filter
    #       e.g. 2-point difference m=2
    #            3-point difference m=3
    # correct = nx1  MTF correction array (limited to a maximum of 10)
    # Example plotted as the MTF (inverse of the correction)
    #  2-point
    correct = np.ones((n, 1))
    m = m-1
    scale = 1
    for i in range(1,n):
        tcorr = np.pi*(i+1)*m/(2*(n+1))
        correct[i] = np.abs(tcorr / np.sin(tcorr))
        correct[i] = 1 + scale*(correct[i]-1)
        if correct[i] > 10:  # Note limiting the correction to the range [1, 10]
            correct[i]  = 10
    return correct

def project(bb, loc, slope, fac):
    # [point, status] = project(bb, loc, slope, fac)
    # Projects the data in array bb along the direction defined by
    #  npix = (1/slope)*nlin.  Used by sfrmat11 and sfrmat2 functions.
    # Data is accumulated in 'bins' that have a width (1/fac) pixel.
    # The smooth, supersampled one-dimensional vector is returned.
    #  bb = input data array
    #  slope and loc are from the least-square fit to edge
    #    x = loc + slope*cent(x)
    #  fac = oversampling (binning) factor, default = 4
    #  Note that this is the inverse of the usual cent(x) = int + slope*xstatus =1;
    #  point = output vector
    #  status = 1, OK
    #  status = 1, zero counts encountered in binning operation, warning is
    #           printed, but execution continues
    # point, status = project(A[:,:,color], loc[color, 1], fitme[color,1], alpha)
    status = 0
    tshape = bb.shape
    nlin = tshape[0]
    npix = tshape[1]

    nn = int(npix * fac)
    slope =  1 / slope
    offset =  int( fac* (0  - (nlin - 1)/slope ) + 0.5)

    del1 = np.abs(offset)
    if offset > 0 :
        offset=0
    barray = np.zeros((2, nn + del1+100))
    # Projection and binning
    for x in range(0,npix):
        for y in range(0,nlin):
            idx =  int(np.ceil((x  - y/slope)*fac) - offset)
            barray[0,idx] = barray[0,idx] + 1
            barray[1,idx]= barray[1,idx] + bb[y,x]

    point = np.zeros((nn,1))
    start = int(0.5*del1 + 0.5) # 四舍五入

    # Check for zero counts
    nz = 0
    for i in range(start,start+nn):
        if barray[0, i] == 0:
            nz = nz + 1
            status = 0
            if i == 0:
                barray[0, 0] = barray[0, 1]
            else:
                barray[0, i] = (barray[0, i - 1]  + barray[0, i + 1] ) / 2
  
        if barray[1, i] == 0:
            nz = nz + 1
            status = 0
            if i==0:
                barray[1, 0] = barray[1, 1] 
            else:
                barray[1, i]  = (barray[1, i - 1]  + barray[1, i + 1] ) / 2

    if status != 0:
        print('WARNING! \n Zero count(s) found during projection binning. The edge angle may be large, or you may need more lines of data Execution will continue, but see Users Guide for info')

    index = np.array(range(start,nn+start),np.int32)
    point = barray[1,index] / barray[0,index]
    return point,status

def sfrfft(x):
    # x 横向量
    xlen = max([x.shape[0],x.shape[1]])
    n = np.arange(0,xlen).reshape(1,xlen)
    k = n.reshape(xlen,1)
    base = np.exp(-1j*2*np.pi/xlen)
    w = np.dot(k,n)
    W = base**w
    XX = np.dot(W,x.T)
    return XX

def calcNyquistFreqMtf(mtfdat,nyquistFreq,mode):
    #奈奎斯特频率Ny = 1/2 * 采样频率
	# Ny/2 = 0.25, Ny/4 = 0.125, Ny/8 = 0.0625
    nyquistFreqMTF = []
    if ( len(nyquistFreq)==0 or mode > 3 or mode < 1):
        return nyquistFreqMTF
	
    if(mode == 2):
        # 双临近点线性插值
        for tSharpness in nyquistFreq:
            delta_freq = np.abs(mtfdat[:,0]-tSharpness)
            position = np.argmin(delta_freq)
            posStart = position
            if(position > 0 and position < mtfdat.shape[0]-1):
                if(tSharpness < mtfdat[position][0]):
                    posStart -= 1
            elif(position == 0):
                posStart = 0
            elif(position == mtfdat.shape[0]-1):
                posStart = position - 1
            slope = (mtfdat[posStart][1] - mtfdat[posStart+1][1]) / (mtfdat[posStart][0]- mtfdat[posStart+1][0]);
            tmpMTF = mtfdat[posStart][1] + (tSharpness - mtfdat[posStart][0]) * slope
            if tmpMTF >= 1:
                tmpMTF = 0.98
            nyquistFreqMTF.append(tmpMTF)
    
    elif(mode == 3):
        #三临近点线性插值
        for tSharpness in nyquistFreq:
            delta_freq = np.abs(mtfdat[:,0]-tSharpness)
            position = np.argmin(delta_freq)
            if(position > 0 and position < mtfdat.shape[0]-1):
                datpart1=mtfdat[position-1:position+2,:]
                posStart = 1.5+0.5*np.sign(tSharpness-mtfdat[position][0]) - 1
                p=np.polyfit(datpart1[:,0],datpart1[:,-1],1)
            elif(position == 0):
                datpart1=mtfdat[position:position+2,:]
                posStart = 0
                p=np.polyfit(datpart1[:,1],datpart1[:,-1],1)     
            elif(position == mtfdat.shape[0]-1):
                datpart1=mtfdat[position-1:position+1,:]  
                posStart = 0
                p=np.polyfit(datpart1[:,1],datpart1[:,-1],1)
            tmpMTF = datpart1[int(posStart)][1]+p[0]*(tSharpness-datpart1[int(posStart)][0])
            if tmpMTF >= 1:
                tmpMTF = 0.98
            nyquistFreqMTF.append(tmpMTF)
    else:
        # 最邻近插值
        for tSharpness in nyquistFreq:
            delta_freq = np.abs(mtfdat[:,0]-tSharpness)
            position = np.argmin(delta_freq)
            tmpMTF = mtfdat[position,1]
            if tmpMTF >= 1:
                tmpMTF = 0.98
            nyquistFreqMTF.append(tmpMTF)

    return nyquistFreqMTF

def sfrmat3(A, io=0, gamma=1, intervalUnit = 1, weight= [0.213,0.715,0.072], oecfname='none'):
    #--------------------------------------------------------------------------
    # [dat1,MTF50,MTF30] = sfrmat3(A, io=0, weight= [0.213,0.715,0.072], gamma=1, intervalUnit = 1, oecfname='none'):
    #  MatLab function: sfrmat3   Slanted-edge Analysis for digital camera and scanner
    #                             evaluation. Updated version of sfrmat2.
    #  [status, dat, fitme, esf, alpha, del2] = sfrmat3(io, intervalUnit, weight, a, oecfname);
    #        From a selected edge area of an image, the program computes
    #        the ISO slanted edge SFR. Input file can be single or
    #        three-record file. Many image formats are supported. The image
    #        is displayed and a region of interest (ROI) can be chosen, or
    #        the entire field will be selected by not moving the mouse
    #        when defining an ROI (simple click). Either a vertical or horizontal
    #        edge features can be analized.
    #  Input arguments:
    #       io  (optional)
    #         0 = (default) R,G,B,Lum SFRs + edge location(s)
    #           = 'sfrmat2'  R,G,B,Lum SFRs + edge location(s)but
    #             with the same calculations as the previous version, sfrmat2
    #         1 = Non GUI usage with supplied data array 不会弹出选择图形区域，直接输入区域数据阵列
    #       intervalUnit (optional) sampling interval in mm or pixels/inch
    #           If dx < 1 it is assumed to be sampling pitch in mm
    #           If io = 1 (see below, no GUI) and del is not specified,
    #           it is set equal to 1, so frequency is given in cy/pixel.
    #       weight (optiona) default 1 x 3 r,g,b weighs for luminance weighting
    #       a   (required if io =1) an nxm or nxmx3 array of data
    #       oecfname  optional name of oecf LUT file containing 3xn or 1xn array
    # 
    #  Returns: 
    #        status = 0 if normal execution
    #        dat = computed sfr data
    #        fitme = coefficients for the linear equations for the fit to
    #                edge locations for each color-record. For a 3-record
    #                data file, fitme is a (4 x 3) array, with the last column
    #                being the color misregistration value (with green as 
    #                reference).
    #        esf = supersampled edge-spread functin array
    #        alpha = binning factor used
    #        del_2 = sampling interval for esf, from which the SFR spatial
    #               frequency sampling is was computed. This will be 
    #               approximately  4  times the original image sampling.
    # 
    # EXAMPLE USAGE:
    #  sfrmat3     file and ROI selection and 
    #  sfrmat3(1) = GUI usage
    #  sfrmat3(0, del) = GUI usage with del as default sampling in mm 
    #                    or dpi 
    #  sfrmat3(2, del, weight) = GUI usage with del as default sampling
    #                    in mm or dpi and weight as default luminance
    #                    weights
    #  sfrmat3(4, dat) = non-GUI usage for data array, dat, with default
    #                    sampling and weights aplied (del =1, 
    #                    weights = [.3 .6 .1])
    #  [status, dat, fitme] = sfrmat3(4, del, weight, a, oecfdat);
    #                    sfr and edge locations, are returned for data
    #                    array dat using oecf array, oecfdat, with
    #                    specified sampling interval and luminance weights
    #  
    # Provided in support of digital imaging performance standards being development
    # by the International Imaging Industry Association (i3a.org).
    # 
    # Author: Peter Burns, 24 July 2009
    #                      12 May 2015  updated legend title to be compatible
    #                      with current Matlab version (legendTitle.m)
    #  Copyright (c) 2009-2015 Peter D. Burns, pdburns@ieee.org
    #--------------------------------------------------------------------------
    #------ 初始化参数 Initialization parameters
    status = 0
    name = 'sfrmat3'
    version = '2.0'
    when = '7 Jue 2023'

    #ITU-R Recommendation  BT.709 weighting
    guidefweight = [0.213,0.715,0.072]
    #Previously used weighting
    #defweight = [0.213   0.715   0.072]
    alpha = 4 # binning, default 4x sampling

    oecfdatflag = 0
    oldflag = 0
    hor2ver = False
    if( len(weight) != 3):
        weight = guidefweight

    if io != 0 and io != 1:
        print(['Input argument io shoud be 0 or 1, setting equal to 0'])
        io = 0
    #------ 初始化参数结束 End of initialization parameters
    #--------------------------
    #------ calc start
    if(A.ndim == 2):
        A = A[:,:,np.newaxis]
    tshape = A.shape
    nlin = tshape[0]
    npix = tshape[1]
    ncol = tshape[2]
    
    # Suppresses interpreting of e.g. filenames
    if io == 1:
        maxVal = A.max()
        if maxVal <= 2^8-1:
            smax = 2^8-1
            A = A.astype(np.uint8)
        elif (maxVal > 2^8-1 and maxVal <= 2^16-1):
            smax = 2^16-1
            A = A.astype(np.uint16)
        else:
            smax = 1e10
            A = A.astype(np.float32)
        # ctype = A.type
        # if(ctype == np.uint8):
        #     smax = 2^8-1
        # elif ctype == np.uint16:
        #     smax = 2^16 - 1
        # else:
        #     smax = 1e10
        # A = A.astype(np.float32)
        [nlow, nhigh, cstatus] = clipping(A, 0, smax, 0.005)

        if oecfname == 'none':
            A = gamma_trans(A,gamma)# gamma校正
        else:
            # Transforms a using OECF LUT from file chosen
            #'Applying OECF look-up table'
            A = A
            #[A, oestatus] = getoecf(A, oecfname); 
        # Assume input was in DPI convert to pitch in mm
        if (intervalUnit > 1):
            intervalUnit = 25.4 / intervalUnit
    #------ END IF(io == 1)
    # Form luminance record using the weight vector for red, green and blue
    if ncol == 3:
        lum = weight[2]*A[:,:,0] + weight[1]*A[:,:,1] + weight[0]*A[:,:,2] # BGR
        lum = lum[:,:,np.newaxis]
        A = np.append(A,lum,axis = 2)
        ncol = 4
    #------ END IF(ncol == 3)
    #--------------------------
    # IF edge is horizontal, Rotate edge so it is vertical
    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 500, 500)
    # cv2.imshow('img',A)
    # cv2.waitKey(0)
    testv = np.abs(A[[0,1,2],:,:].mean() - A[[-3,-2,-1],:].mean())
    testh = np.abs(A[:,[0,1,2],:].mean() - A[:,[-3,-2,-1],:].mean())
    if testv > testh:
        hor2ver = True
        # for i in range(0,ncol):
        #     tmp = A[:,:,i].T
        #     #tmp = np.flip(tmp,axis = 0)
        #     A[:,:,i] = tmp
        t = nlin
        nlin = npix
        npix = t
        A = A.T
        A = A.reshape(nlin,npix,ncol)
        A = np.flip(A,axis = 0)

    # cv2.namedWindow("img", 0)
    # cv2.resizeWindow("img", 500, 500)
    # cv2.imshow('img',A)
    # cv2.waitKey(0)
    #------ END IF(edge is horizontal)
    #--------------------------
    #------ start ployfit edge
    loc = np.zeros((ncol, nlin))
    fil1 = (0.5,-0.5)
    fil2 = (0.5,0,-0.5)
    # We Need 'positive' edge
    tleft = A[:,[0,1,2,3,4],0].sum()
    tright = A[:,[-5,-4,-3,-2,-1],0].sum()
    if tleft > tright:
        fil1 = (-0.5,0.5)
        fil2 = (-0.5,0,0.5)
    # Test for low contrast edge;
    test = abs(np.float32(tleft) - np.float32(tright)) / (np.float32(tleft) + np.float32(tright))
    if test < 0.2:
        print(' ** WARNING: Edge contrast is less that 20%, this can lead to high error in the SFR measurement.')

    fitme = np.zeros((ncol, 2),np.float32)
    slout = np.zeros((ncol, 1),np.float32)

    # Smoothing window for first part of edge location estimation to be used on each line of ROI
    win1 = apply_hamming_window(npix, (npix+1)/2)    # Symmetric window

    # Loop for each color
    yIndex = np.array(range(0,nlin),np.float32) # 竖直斜线的拟合 创建 y index;
    for color in range(0,ncol):
        c = deriv1(A[:,:,color], nlin, npix, fil1)
        # 滤波处理，凸显边缘
        # compute centroid for derivative array for each line in ROI. SET WINDOW array as 'win'
        for r in range(0,nlin):
            loc[color, r] = centroid( c[r,:]*win1) - 0.5    # -0.5 shift for FIR phase
        
        #-- 找到边缘质心位置
        #-- 一阶线性拟合 ployfit ：x = int + slope*cent(x) 得出初始值和斜率；
        xShift = np.array(loc[color, :])
        points = np.vstack((yIndex,xShift)).T # x y 颠倒再拟合，预防 k 不存在
        output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        # (x-x0) / vx = (y-y0) / vy
        # (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line
        # (x0, y0) is a point on the line
        k = output[1] / (output[0]+eps)
        b = output[3] - k * output[2]
        #------ END OF first ployfit
        #-- 线性回归，再次滤波，并质心线性拟合，目的减少噪声对定位精度的影响；
        newX = k * yIndex + b
        for r in range(0,nlin):
            win2 = apply_hamming_window(npix, newX[r])
            loc[color, r] = centroid( c[r,:] * win2) - 0.5 # -0.5 shift for FIR phase
        
        
        xShift = np.array(loc[color, :])
        points = np.vstack((yIndex,xShift)).T # x y 颠倒再拟合，预防 k 不存在
        output = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        # (x-x0) / vx = (y-y0) / vy
        # (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line
        # (x0, y0) is a point on the line
        fitme[color,0]  = output[1] / (output[0]+eps)
        fitme[color,1]  = output[3] - k * output[2]
        #------ END OF second ployfit
        #------ check io > 0
        if io > 0:
            slout[color] = - 1./fitme[color,1]    # slope is as normally defined in image coods.
            if hor2ver:                   # positive flag it ROI was rotated
                slout[color] = -fitme[color,1]
        #------ end check io > 0
    #------ End of loop for each color 
    #------ end ployfit edge
        #--------------------------                     
        # Full linear fit is available as variable fitme. Note that the fit is for the projection onto the X-axis,
        #       x = fitme(color, 1) y + fitme(color, 2)
        # so the slope is the inverse of the one that you may expect
        # Limit number of lines to integer(npix*line slope as per ISO algorithm except if processing as 'sfrmat2'
        #--------------------------
        #------ check oldflag
        if oldflag != 1:
            nlin1 = int( np.floor( nlin * np.abs(fitme[0,0] ) * 100) / ( 100 * np.abs( fitme[0,0] )+eps ) + 0.5)
            if nlin1 < 2:
                nlin1 = nlin
            if ( nlin1 == np.nan or nlin1 == np.inf ) :
                dat1 = 0
                MTF50 = 0
                MTF30 = 0
                return dat1,MTF50,MTF30
            A = A[0:nlin1, :, 0:ncol]
        #------ end check oldflag
        slope_deg = 180 * np.arctan(abs(fitme[0,0])) / np.pi
        # 180/pi将数值转换成角度，pi=180度
        if slope_deg < 3.5:
            print(['High slope warning ',str(slope_deg),' degrees'])
        
        nn  = int(npix * alpha)
        mtf = np.zeros((nn, ncol))
        nn2 = int(nn/2) + 1

        if oldflag != 1:
            #Correct sampling inverval for sampling parallel to edge
            delfac = np.cos(np.arctan(fitme[0,0]))
            intervalUnit = intervalUnit * delfac
            # Derivative correction
            dcorr = fir2fix(nn2, 3) # dcorr corrects SFR for response of FIR filter,该修正系数如何得到的？

        freq = (alpha / intervalUnit / nn) * np.array(range(0,nn),np.float32)

        freqlim = 1
        if alpha == 1:
            freqlim = 2
        nn2out = int(nn2*freqlim/2+0.5)
        win = apply_hamming_window(nn,(nn+1)/2) # centered Hamming window
        #------ Large SFR loop for each color record
        esf = np.zeros((nn,ncol)) 

        for color in range(0,ncol):
            #-- project and bin data in 4x sampled array
            point, status = project(A[:,:,color], loc[color, 1], fitme[color,0], alpha)

            esf[:,color] = point
            # ESF边缘扩展函数；
            # compute first derivative via FIR (1x3) filter fil
            # ---Modified Mar,28,2017----
            c = deriv1(np.array([point,]), 1, nn, fil2)
            mid = centroid(c)
            temp = cent(c, int(mid+0.5))  #shift array so it is centered
            ansfft = np.abs(sfrfft(temp * win)) # ansfft 列向量
            mtf[0:nn2, color] = ansfft[0:nn2,0] / ansfft[0,0]  #归一化处理
            if oldflag !=1:
                mtf[0:nn2, color] = np.squeeze(np.reshape(mtf[0:nn2, color], (-1,1)) * dcorr) #MTF校正
    #----------------------------------------------------------
    dat = np.zeros((nn2out, ncol+1))
    for i in range(0,nn2out):
        dat[i,:] = np.squeeze([freq[i],np.squeeze(mtf[i,:])])

    if ncol == 4:
        dat1 = dat[:,[1,5]]
    else:
        dat1 = dat

    #MTF50
    tmp1 = np.abs(dat1[:,0]-0.6)
    positionFreqCut = np.where(tmp1 == np.min(tmp1)) #cutoff of frequence
    tmp1 = np.abs(dat1[0:positionFreqCut[0][0]+1,-1]-0.5)
    position = np.where(tmp1 == np.min(tmp1))

    posStart = int(0)
    minPos = position[0][0]
    MTF50 = 0
    if (minPos > 0 and minPos < positionFreqCut[0][0]+1):
        datpart1 = dat1[minPos-1:minPos+2,:]
        posStart = int(1.5 + 0.5 * np.sign(dat1[minPos,-1]-0.5) - 1)
    elif (minPos == 0):
        datpart1 = dat1[[0,1],:]
    elif (minPos == positionFreqCut[0][0]+1):
        datpart1 = dat1[[-2,-1],:]
    
    output = cv2.fitLine(datpart1[:,[1,0]], cv2.DIST_L2, 0, 0.01, 0.01)
    k = output[1] / (output[0]+eps)
    b = output[3] - k * output[2]
    MTF50 = datpart1[posStart,0] + k *(0.5-datpart1[posStart,-1])
    #MTF50=p(2)+p(1)*0.5;

    #MTF30
    tmp1 = np.abs(dat1[0:positionFreqCut[0][0]+1,-1]-0.2)
    position = np.where(tmp1 == np.min(tmp1))
    minPos = position[0][0]
    posStart = int(0)
    MTF30 = 0
    if (minPos > 0 and minPos < positionFreqCut[0][0]+1):
        datpart1 = dat1[minPos-1:minPos+2,:]
        posStart = int(1.5 + 0.5 * np.sign(dat1[minPos,-1]-0.2) - 1)
    elif (minPos == 0):
        datpart1 = dat1[[0,1],:]
    elif (minPos == positionFreqCut[0][0]+1):
        datpart1 = dat1[[-2,-1],:]
    
    output = cv2.fitLine(datpart1[:,[1,0]], cv2.DIST_L2, 0, 0.01, 0.01)
    k = output[1] / (output[0]+eps)
    b = output[3] - k * output[2]
    MTF30 = datpart1[posStart,0] + k *(0.2-datpart1[posStart,-1])

    return dat1,MTF50[0],MTF30[0]



if __name__ == '__main__':
    img0 = cv2.imread(r'D:\tem\Temp\SFR\fail_roi.png',0)

    dat1,MTF50,MTF30 = sfrmat3(img0, io=1, weight= [0.2126,0.7152,0.0722], gamma=1, intervalUnit = 1, oecfname='none')
    
    print(MTF50," ",MTF30,"\n")
    version = 3
    # # show image
    # imgShow = cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    # pts = Line2Point(output,(60,60))
    # cv2.line(imgShow,(round(pts[0][0]),round(pts[0][1])),(round(pts[1][0]),round(pts[1][1])),(0,255,255))
    # cv2.namedWindow('show',cv2.WINDOW_NORMAL)
    # cv2.imshow('show',imgShow)
    # cv2.waitKey(0)
    pass