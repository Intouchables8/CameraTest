#include "sfrmat3.h"

sfrmat3::sfrmat3(float _gamma, double _intervalUnit, std::vector<double> _weight, std::string _oecfname) {
	this->SetParams( _gamma, _intervalUnit, _weight,_oecfname);
}

std::string sfrmat3::versionInfo()
{
	return "name:sfrmat3,base:ITU-R BT.709,version:3.1,when:20240607";
}

void sfrmat3::SetParams(float _gamma, double _intervalUnit, std::vector<double> _weight, std::string _oecfname)
{
	this->gamma = _gamma;
	this->intervalUnit = _intervalUnit;
	this->oecfname = _oecfname;
	// ITU-R Recommendation  BT.709 weighting
	if (_weight.size() < 3) {
		this->weight = std::vector<double>({ 0.213, 0.715, 0.072 });
	}
	else
	{
		this->weight = _weight;
	}
	//------ Assume input was in DPI convert to pitch in mm
	if (_intervalUnit > 1) {
		this->intervalUnit = 25.4 / _intervalUnit;
		this->funit = "cy/mm";
	}
	else if (this->intervalUnit == 1) {
		this->funit = "cy/pixel";
	}
	else {
		this->funit = "cy/mm";
	}
}

cv::Mat sfrmat3::gammaTransform(cv::Mat& srcImage, float kFactor)
{
	unsigned char LUT[256]{};
	for (int i = 0; i < 256; i++)
	{
		float f = (i + 0.5f) / 255;
		f = (float)(pow(f, kFactor));
		LUT[i] = cv::saturate_cast<uchar>(f * 255.0f - 0.5f);
	}
	cv::Mat resultImage = srcImage.clone();
	if (srcImage.channels() == 1)
	{
		cv::MatIterator_<uchar> iterator = resultImage.begin<uchar>();
		cv::MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
		for (; iterator != iteratorEnd; iterator++)
		{
			*iterator = LUT[(*iterator)];
		}
	}
	else
	{
		cv::MatIterator_<cv::Vec3b> iterator = resultImage.begin<cv::Vec3b>();
		cv::MatIterator_<cv::Vec3b> iteratorEnd = resultImage.end<cv::Vec3b>();
		for (; iterator != iteratorEnd; iterator++)
		{
			(*iterator)[0] = LUT[((*iterator)[0])];//b
			(*iterator)[1] = LUT[((*iterator)[1])];//g
			(*iterator)[2] = LUT[((*iterator)[2])];//r
		}
	}
	return resultImage;
}

cv::Mat sfrmat3::matRotateCounterClockWise90(cv::Mat src)
{
	if (src.empty())
	{
		//std::cout << "RorateMat src is empty!";
	}
	// 矩阵转置
	cv::transpose(src, src);
	//0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
	cv::flip(src, src, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	return src;
}

cv::Mat sfrmat3::rotatev2(cv::Mat img,bool & hor2ver)
{
	cv::Mat v1 = img.rowRange(0, 3);
	cv::Mat v2 = img.rowRange(img.rows-3, img.rows);
	double testv = cv::abs(cv::mean(v1).val[0] - cv::mean(v2).val[0]);

	cv::Mat h1 = img.colRange(0,3);
	cv::Mat h2 = img.colRange(img.cols-3,img.cols);
	double testh = cv::abs(cv::mean(h1).val[0] - cv::mean(h2).val[0]);

	hor2ver = false;
	cv::Mat output = img.clone();
	if (testv > testh)
	{
		//std::cout << "SFR srcImage rotate 90 degree" << std::endl;
		hor2ver = true;
		output = matRotateCounterClockWise90(img);
	}
	return output;
}

//generates a general asymmetric Hamming-type window array
//If mid = (n+1)/2 then the usual symmetric Hamming window is returned
cv::Mat sfrmat3::ahamming(int n, double mid)
{
	cv::Mat hammingData = cv::Mat::zeros(1,n, CV_64FC1);
	double wid1 = mid - 1;
	double wid2 = n - mid;
	double wid = max(wid1, wid2);
	auto* ptr = hammingData.ptr<double>(0);
	for (int i = 0; i < n; i++)
	{
		double arg = (i - mid + 1) / wid;
		ptr[i] = 0.54 + 0.46 * cosf(CV_PI * arg);
	}
	return hammingData;
}

// Computes first derivative via FIR(1xn) filter
//  Edge effects are suppressed and vector size is preserved
//  Filter is applied in the nHeight direction only
//   a = (nWidth, nHeight) data array
//   fil = array of filter coefficients, eg[-0.5 0.5]
cv::Mat sfrmat3::derivationFun(const cv::Mat& input, cv::Mat fil)
{
	int nRows = input.rows;
	int nCols = input.cols;
	//int nn = fil.cols;
	cv::Mat der_mat, fil_conv;
	cv::flip(fil, fil_conv, 1); // Convolutional need flip kernel,while filter do not

	cv::filter2D(input, der_mat, fil.type(), fil_conv, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	der_mat.col(1).copyTo(der_mat.col(0));
	return der_mat;
}

double sfrmat3::centroid(const cv::Mat& input)
{
	// input 1*n 
	int n = input.cols;
	double sumx = cv::sum(input).val[0];
	auto* ptr = input.ptr<double>();
	double loc = 0;
	if (sumx > 1e-4) {
		for (int i = 0; i < n; ++i) {
			loc += (ptr[i] * (i + 1) / sumx);
		}
		//loc /= sumx;//-0.5 shift for FIR phase
	}
	return loc;
}

std::vector<double> sfrmat3::polyfit(std::vector<cv::Point2d>& in_point, int n)
{
	int size = in_point.size();
	//所求未知数个数
	int x_num = n + 1;
	//构造矩阵U和Y
	cv::Mat mat_u(size, x_num, CV_64F);
	cv::Mat mat_y(size, 1, CV_64F);

	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}

	for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}

	//矩阵运算，获得系数矩阵K
	cv::Mat mat_k(x_num, 1, CV_64F);
	mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;// [b,k]
	return { mat_k.at<double>(1, 0), mat_k.at<double>(0, 0) };//[k,b]
}

/* Correction for MTF of derivative (difference) filter
* n = frequency data length [0-half-sampling (Nyquist) frequency]
* m = length of difference filter
*      e.g. 2-point difference m=2
*            3-point difference m=3
* correct = 1xN  MTF correction array (limited to a maximum of 10)
*
* Example plotted as the MTF (inverse of the correction)
*/
cv::Mat sfrmat3::fir2fix(int n, int m)
{
	//std::cout << "Derivative correction " << std::endl;
	cv::Mat correct = cv::Mat::ones(1,n, CV_64FC1);

	m = m - 1;
	int scale = 1;
	auto ptr = correct.ptr<double>();
	for (int i = 1; i < n; i++)
	{
		double tmp = CV_PI * (i + 1) * m / (2 * (n + 1));
		ptr[i] = abs(tmp) / sinf(tmp);
		ptr[i] = 1 + scale * (ptr[i] - 1);
		if (ptr[i] > 10) //limiting the correction to the range[1, 10]
		{
			ptr[i] = 10;
		}
	}
	return correct;
}

/*
* Projects the data in array bb along the direction defined by
*  npix = (1/slope)*nlin.  Used by sfrmat11 and sfrmat2 functions.
* Data is accumulated in 'bins' that have a width (1/fac) pixel.
* The smooth, supersampled one-dimensional vector is returned.
*  bb = input data array
*  slope and loc are from the least-square fit to edge
*    x = loc + slope*cent(x)
*  fac = oversampling (binning) factor, default = 4
*  Note that this is the inverse of the usual cent(x) = int + slope*xstatus =1;
*  point = output vector
*  status = 1, OK
*  status = 1, zero counts encountered in binning operation, warning is
*           printed, but execution continues
*
*/
cv::Mat sfrmat3::project(const cv::Mat& bb, double loc, double slope, int fac)
{
	int nWidth = bb.rows;
	int nHeight = bb.cols;
	int big = 0;
	int nn = nHeight * fac;

	// smoothing window
	cv::Mat win = ahamming(nn, fac * loc);

	slope = 1 / slope;

	int offset = round(fac * (0 - (nWidth - 1) / slope));
	int del = abs(offset);
	if (offset > 0)
	{
		offset = 0;
	}

	cv::Mat barray = cv::Mat::zeros(2, nn + del + 100, CV_64FC1);
	auto* ptr_barray_1 = barray.ptr<double>(0);
	auto* ptr_barray_2 = barray.ptr<double>(1);

	for (int n = 0; n < nHeight; n++)
	{
		for (int m = 0; m < nWidth; m++)
		{
			int x = n;
			int y = m;
			int ling = ceil((x - y / slope) * fac) - offset;
			ptr_barray_1[ling] = ptr_barray_1[ling] + 1;
			ptr_barray_2[ling] = ptr_barray_2[ling] + bb.at<uchar>(m, n);
		}
	}
	int start = round(0.5 * del);
	int nz = 0;
	int status = 0;
	for (int i = start; i < start + nn; i++)
	{
		if (ptr_barray_1[i] == 0)
		{
			nz++;
			status = 0;
			if (i == 1)
			{
				ptr_barray_1[i] = ptr_barray_1[i + 1];
			}
			else
			{
				ptr_barray_1[i] = (ptr_barray_1[i - 1] + ptr_barray_1[i + 1]) / 2.0;
			}
		}
		if (ptr_barray_2[i] == 0)
		{
			nz++;
			status = 0;
			if (i == 1)
			{
				ptr_barray_2[i] = ptr_barray_1[2 + 1];
			}
			else
			{
				ptr_barray_2[i] = (ptr_barray_2[i - 1] + ptr_barray_2[i + 1]) / 2.0;
			}
		}
	}

	cv::Mat point = cv::Mat::zeros(1, nn, CV_64FC1);
	auto* ptr_point = point.ptr<double>();
	for (int i = 0; i < nn; i++)
	{
		ptr_point[i] = ptr_barray_2[i + start] / ptr_barray_1[i + start];
	}

	return point;
}

/*
* Matlab function cent, shift of one-dimensional array, so that a(center) is located at b(round((n+1)/2).
* Written to shift a line-spread function array prior to applying a smoothing window.
*  a      = input array
*  center = location of signal center to be shifted
*  b      = output shifted array
*/
cv::Mat sfrmat3::cent(const cv::Mat& a, int center)
{
	int n = a.cols;
	cv::Mat b = cv::Mat::zeros(1,n, a.type());
	int mid = round((n + 1) / 2.0);
	int del = round(center - mid);
	auto ptr_a = a.ptr<double>();
	auto ptr_b = b.ptr<double>();
	if (del > 0)
	{
		for (int i = 0; i < n - del; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}
	else
	{
		for (int i = -del; i < n; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}

	return b;
}

cv::Mat sfrmat3::complex_abs(const cv::Mat& input)
{
	std::vector<cv::Mat> channels;
	cv::split(input, channels);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	cv::Mat output = cv::Mat::zeros(1,input.rows, CV_64FC1);
	auto ptr_output = output.ptr<double>(0);
	for (int i = 0; i < input.rows; i++)
	{
		double a = ptr_1[i];
		double b = ptr_2[i];
		ptr_output[i] = sqrtf(a * a + b * b);
	}
	return output;
}

void sfrmat3::sampeff(double& eff, std::vector<double>& freqval, std::vector<double>& sfrval, const cv::Mat& mtfout, const std::vector<double>& valList, double del)
{
	//valList 必须从小到大排列
	int fflag = 0;
	int nchannel = mtfout.rows - 1;
	int nval = valList.size();
	double delf = mtfout.at<double>(0, 1) + 1e-6;
	double hs = 0.5 / del;
	auto* ptr_freq = mtfout.ptr<double>(0);
	int x_freqMin = -1, x_mtfMin = -1;
	for (int i = 0; i < mtfout.cols; ++i) {
		if (ptr_freq[i] > 1.1 * hs){
			x_freqMin = i;
			break;
		}
	}
	if (x_freqMin == -1)
	{
		x_freqMin = mtfout.cols;
		x_mtfMin = x_freqMin;
	}
	else
	{
		for (int i = 0; i < mtfout.cols; ++i) {
			if (ptr_freq[i] > hs - delf){
				x_mtfMin = i;
				break;
			}
		}
	}
	cv::Mat dat_new = mtfout( cv::Rect(0, 0, x_freqMin, mtfout.rows) ).clone();
	freqval = std::vector<double>(nval, 0);
	sfrval = std::vector<double>(nval, 0);
	eff = 0;
	
	if (fflag) {
		cv::blur(dat_new, dat_new, cv::Size(3, 1), cv::Point(-1, -1), cv::BORDER_REPLICATE);
	}

	double maxf = ptr_freq[x_freqMin - 1];

	double* ptr_mtf = dat_new.ptr<double>(1);
	double freqval_temp = 0;
	double sfr_val_temp = 0;
	for (int j = 0; j < nval; j++)
	{
		int xx = -1;
		double val = valList[j];
		for (int k = 0; k < dat_new.cols; ++k) {
			// First crossing of threshold
			if (ptr_mtf[k] < val)
			{
				xx = k-1;
				break;
			}
		}
		if (xx < 1) {
			freqval_temp = maxf;
			sfr_val_temp = ptr_mtf[x_freqMin - 1];
		}
		else {
			//interpolation
			double slop_tmp = (ptr_mtf[xx + 1] - ptr_mtf[xx]) / ptr_freq[1];
			double dely = ptr_mtf[xx] - val;
			freqval_temp = ptr_freq[xx] - dely / slop_tmp;
			sfr_val_temp = ptr_mtf[xx] - dely;
		}
		if (freqval_temp > maxf) {
			freqval_temp = maxf;
			sfr_val_temp = ptr_mtf[x_freqMin - 1];
		}
		freqval[j] = freqval_temp;
		sfrval[j] = sfr_val_temp;
	}

	//Efficiency computed only for lowest value of SFR requested
	int idx = 0,ii=0;
	double mmin = 1;
	for (auto mm:valList) {
		if (mmin > mm)
		{
			mmin = mm;
			idx = ii;
		}
		ii++;
	}
	eff = min(round(100 * freqval[idx] / ptr_freq[x_mtfMin - 1]), 100);
}

void sfrmat3::showChart(const cv::Mat& mtfout)
{
	//打印输出
	cv::Mat showImg = cv::Mat(150, 250, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < mtfout.cols; i++)
	{
		std::cout << i << ": " << mtfout.col(i).t() << std::endl;
		int x = mtfout.at<double>(0, i) * 100 + 10;
		int y = mtfout.at<double>(1, i) * 100 + 10;
		showImg.at<cv::Vec3b>(y, x) = cv::Vec3b({ 0,0,0 });
	}
	cv::arrowedLine(showImg, cv::Point(10, 10), cv::Point(210, 10), cv::Scalar(255, 255, 0), 1, 4, 0, 0.02);
	cv::arrowedLine(showImg, cv::Point(10, 10), cv::Point(10, 130), cv::Scalar(255, 255, 0), 1, 4, 0, 0.02);
	cv::flip(showImg, showImg, 0);
	cv::putText(showImg, "freq", cv::Point(210, 145), cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255, 255, 0));
	cv::putText(showImg, "MTF", cv::Point(2, 15), cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(255, 255, 0));

	cv::namedWindow("showMTF", cv::WINDOW_NORMAL);
	cv::imshow("showMTF", showImg);
	cv::waitKey(0);
}

int sfrmat3::calcSingleChannelMTF(cv::Mat& mtfout, std::vector<double>& fitCoeff, const cv::Mat& srcimg)
{
	// mtfout     [freqList;lum_mtfList]  2xN                     
	// fitCoeff   [ lum_k,lum_b]          2x1
	// mtf3050    [ lum_mtf30,lum_mtf50]  2x1

	/* intervalUnit(optional) sampling interval in mm or pixels / inch
	 * ** If intervalUnit < 1 it is assumed to be sampling pitch in mm
	 * ** If intervalUnit > 1 it is assumed to be dpi (Dots Per Inch)
	 * ** If intervalUnit = 1 so frequency is given in cy / pixel, if intervalUnit is not specified,it is set equal to 1.*/

	//------ Initialization parameter
	int nbin = 4; // binning, default 4x sampling
	double _intervalUnit = this->intervalUnit;
	bool hor2ver = false;
	//----------------------------------------------------------
	//--- IF edge is horizontal, Rotate edge so it is vertical
	cv::Mat img = rotatev2(srcimg, hor2ver);
	if (abs(gamma - 1.0) > 0.01)
	{
		//std::cout << "SFR gammaTransform" << std::endl;
		img = gammaTransform(img, gamma);
	}
	//--------------------------------------------------
	int img_height = img.rows;//nlin
	int img_width = img.cols;//npix
	int img_channels = img.channels(); // ncol
	//--- deriv1 kernel
	cv::Mat fil1 = (cv::Mat_<double>(1, 2) << 0.5, -0.5);
	cv::Mat fil2 = (cv::Mat_<double>(1, 3) << 0.5, 0, -0.5);
	//--- is left(row * 5) or right(row * 6) 
	double tleft = cv::sum(img.colRange(0, 5)).val[0];
	double tright = cv::sum(img.colRange(img_width - 6, img_width)).val[0];
	double test = (tleft - tright) / (tleft + tright);
	if (tleft > tright)
	{
		fil1 = (cv::Mat_<double>(1, 2) << -0.5, 0.5);
		fil2 = (cv::Mat_<double>(1, 3) << -0.5, 0, 0.5);
	}
	if (test < 0.2 && test > -0.2)
	{
		std::cout << "** WARNING: Edge contrast is less that 20%, this can lead to high error in the SFR measurement." << std::endl;
	}
	//--------------------------------------------------
	//汉明窗
	cv::Mat win1 = ahamming(img_width, (img_width + 1) / 2.0);
	//求一阶导数
	cv::Mat derivMat = derivationFun(img, fil1);
	//计算质心
	std::vector<cv::Point2d> fit_centroids(img_height, cv::Point2d());
	for (int i = 0; i < img_height; i++)
	{
		cv::Mat temp = derivMat.row(i);// don`t change the value of temp,if need, please add .clone()
		double centroid_x = centroid( temp.mul(win1)) - 0.5;
		fit_centroids[i] = cv::Point2d(i, centroid_x);
	}
	//--- 曲线拟合 
	//--- 由于斜边图是竖直的，预防拟合时出现 k 不存在的情况,所以拟合时将 x y 颠倒;
	fitCoeff = polyfit(fit_centroids, 1);
	double fitme_k = fitCoeff[0];
	double fitme_b = fitCoeff[1];
	for (int i = 0; i < img_height; i++)
	{
		double placeVal = fitme_k * (i + 1) + fitme_b;
		cv::Mat win2 = ahamming(img_width, placeVal);
		cv::Mat temp = derivMat.row(i);// don`t change the value of temp,if need, please add .clone()
		double centroid_x = centroid(temp.mul(win2));
		fit_centroids[i] = cv::Point2d(i, centroid_x);
	}
	fitCoeff = polyfit(fit_centroids, 1);
	//Limit number of lines to integer
	//对应oldflag=0
	fitme_k = fitCoeff[0];
	int newH1 = round(floor(img_height * abs(fitme_k) * 100) / (100 * abs(fitme_k)));
	//int newH1 = round(img_height * cv::abs(fitme_k) / cv::abs(fitme_k));
	// -----------20240607 新增在newH1为0时,无数据后续会报错,新增判断 -----------
	if (newH1 < 2)
	{
		newH1 = img_height;
	}
	// ---------------------------------------------------------------------------
	cv::Rect interger_roi(0, 0, img_width, newH1);
	cv::Mat img_roi_new = img(interger_roi).clone();
	//----
	_intervalUnit *= cosf(atan(fitme_k));
	double slope_deg = 180 * atan(cv::abs(fitme_k)) / CV_PI;
	if (slope_deg < 3.5)
	{
		//std::cout << "High slope warning : " << slope_deg << " degrees" << std::endl;
	}
	//Correct sampling inverval for sampling parallel to edge
	//对应oldflag=0
	int nwbin = img_width * nbin;// nbin * width -->n num, w width ; bin nbin
	int nwbin2d = nwbin / 2 + 1; // 2d-> devide 2 
	int freqlim = (1 == nbin) ? 2 : 1;
	int nn2out = int(nwbin2d * freqlim / 2.0 + 0.5);
	
	// mtf and freq
	cv::Mat mtf = cv::Mat::zeros(1, nwbin, CV_64FC1);
	cv::Mat freq = cv::Mat::zeros(1, nwbin, CV_64FC1);
	//dcorr corrects SFR for response of FIR filter
	cv::Mat dcorr = fir2fix(nwbin2d, 3);
	auto ptr_freq = freq.ptr<double>(0);
	double tmp_c = 1 / (_intervalUnit * img_width + EPS);
	for (int i = 0; i < nwbin; i++)
	{
		ptr_freq[i] = i * tmp_c;
	}
	//double nfreq = nwbin / (2.0 * _intervalUnit * nwbin); // half - sampling frequency
	cv::Mat win3 = ahamming(nwbin, (nwbin + 1) / 2.0);

	// ESF project and bin data in 4x sampled array
	cv::Mat esf = project(img_roi_new, fit_centroids[0].y, fitme_k, nbin);
	//compute first derivative via FIR(1x3) filter fil
	cv::Mat psf = derivationFun(esf, fil2);
	double mid = centroid(psf);
	psf = cent(psf, round(mid));
	// apply window(symmetric Hamming)
	psf = win3.mul(psf);
	//Transform, scale and correct for FIR filter response
	//temp = abs(fft(c, nwbin));
	cv::Mat temp;
	cv::dft(psf.t(), temp, cv::DFT_COMPLEX_OUTPUT);
	temp = complex_abs(temp);
	cv::Rect roi_temp(0, 0, nwbin2d, 1);
	cv::Mat temp1 = temp(roi_temp).clone();
	temp1 = temp1 / temp1.at<double>(0, 0);
	cv::Mat mtf_roi = mtf(roi_temp);
	temp1.copyTo(mtf_roi);
	//对应oldflag=0
	mtf_roi = mtf_roi.mul(dcorr);

	cv::Mat freq0 = freq(cv::Rect(0, 0, nn2out, 1)).clone();
	cv::Mat mtf0 = mtf(cv::Rect(0, 0, nn2out, 1)).clone();
	cv::vconcat(freq0, mtf0, mtfout);
	return 0;
}

std::vector<double> sfrmat3::calcMTFrdPercentFreq(const cv::Mat& mtfout, const std::vector<double>& fitCoeff,std::vector<double> mtfPeakrdPercent)
{
	// mtfPeakrdPercent: MTF peak reduction percent
	// MTF50是当MTF数值下降至最大值的50%时，对应的频率(Cycle Per Pixel)
	// MTF数值的最大值的50%对应的频率值。其中MTF50P一般会使用LW/PH作为单位，LP/PH= Cycle Per Pixel * Total Pixel * 2.
	/* 以下是MTF50P的算法：此算法来源于ISO12233中标准的斜边，并使用了部分数学算法辅助完成。
    *1. 使用OECF Chart 或者 Gamma 转换来改变由影像模组加上的 Gamma（一般影像模组均采用0.5）。
    *2. 分别计算R,G,B,Y四个频道的每个扫描线的此点与前面点的差值，并找到数值差异最大的位置。
    *3. 分别对R,G,B,Y四个频道的差异最大的点组成的曲线作线性回归，但是由于Lens的Distortion的影响，我们需要作二次曲线拟合。
    *4. 通过了曲线拟合，我们产生了四条平均曲线，我们分别取这四条曲线的的小数部分按1/4向下取整。
    *5. 通过第4步，我们产生了重新采样的四条曲线，这四条曲线满足采样定理。
    *6. 计算这四条曲线的相邻点的差分并使用汉明窗函数使用这四条微分曲线的终点的微分数值置0。
    *7. 对这四条微分曲线做快速付立叶变换就可以得到MTF曲线了。
	*/
	//-------------------------------------------
	//Sampling efficiency
	if (mtfPeakrdPercent.empty()) {
		mtfPeakrdPercent.push_back(0.5);
	}
	std::vector<double> MTFrdPercentFreq, sfrval;
	double eff;
	double del = this->intervalUnit * cosf(atan(fitCoeff[0]));
	sampeff(eff, MTFrdPercentFreq, sfrval, mtfout, mtfPeakrdPercent, del);

	bool showflag = false;// Plot SFRs on same axes
	if (showflag) {
		showChart(mtfout);
	}
	return MTFrdPercentFreq;
}

std::vector<double> sfrmat3::calcNyquistFreqMTF(const cv::Mat& mtfout, const std::vector<double>& NyquistFreq, int mode)
{
	//奈奎斯特频率Ny = 1/2 * 采样频率
	// Ny/2 = 0.25, Ny/4 = 0.125, Ny/8 = 0.0625
	std::vector<double> NyquistFreqMTF;
	if (NyquistFreq.empty() || mode > 3 || mode < 1) return NyquistFreqMTF;

	cv::Mat mtf_tmp = mtfout.clone();
	double* ptr_freq = mtf_tmp.ptr<double>(0);
	std::vector<int> min_idx(NyquistFreq.size(), -1);
	if (mode == 2) {
		// 临近点线性插值
		for (int idx0 = 0; idx0 < NyquistFreq.size(); ++idx0) {
			int idx1 = 1;// 0 不参与对比
			for (; idx1 < mtfout.cols; ++idx1) {
				if (ptr_freq[idx1] > NyquistFreq[idx0]) break;
			}
			min_idx[idx0] = idx1;
		}

		for (int i = 0; i < mtfout.rows - 1; ++i) {
			double* ptr_mtf = mtf_tmp.ptr<double>(i + 1);
			int j = 0;
			for (int idx : min_idx) {
				double slope = (ptr_mtf[idx] - ptr_mtf[idx - 1]) / (ptr_freq[idx] - ptr_freq[idx - 1]);
				double tt = ptr_mtf[idx] - (ptr_freq[idx] - NyquistFreq[j]) * slope;
				NyquistFreqMTF.push_back((tt < 1 ? tt : 0.98));
				j++;
			}
		}
	}
	else if (mode == 3) {
		// 三临近点线性插值
		for (int idx0 = 0; idx0 < NyquistFreq.size(); ++idx0) {
			int idx1 = 1;// 0 不参与对比
			for (; idx1 < mtfout.cols; ++idx1) {
				if (ptr_freq[idx1] > NyquistFreq[idx0]) break;
			}
			if (abs(ptr_freq[idx1] - NyquistFreq[idx0]) > abs(ptr_freq[idx1 - 1] - NyquistFreq[idx0])) {
				idx1 -= 1;
			}
			min_idx[idx0] = idx1;
		}

		for (int i = 0; i < mtfout.rows - 1; ++i) {
			double* ptr_mtf = mtf_tmp.ptr<double>(i + 1);
			int j = 0;
			double tt = 0;
			for (int idx : min_idx) {
				int posStart = 0;
				if (idx > 0 && idx < mtfout.cols - 1) {
					std::vector<cv::Point2d> points(3, cv::Point2d());
					tt = ptr_freq[idx] - NyquistFreq[j];
					posStart = (tt > 0 ? 0 : 1);
					points[0] = cv::Point2d(ptr_freq[idx], ptr_mtf[idx]);
					points[1] = cv::Point2d(ptr_freq[idx - 1], ptr_mtf[idx - 1]);
					points[2] = cv::Point2d(ptr_freq[idx + 1], ptr_mtf[idx + 1]);
					std::vector<double> fit_tmp = polyfit(points, 1);
					tt = ptr_mtf[idx + posStart] - tt * fit_tmp[0];
				}
				else if (idx == 0) {
					double slope = (ptr_mtf[1] - ptr_mtf[0]) / (ptr_freq[1] - ptr_freq[1]);
					tt = ptr_mtf[1] - (ptr_freq[1] - NyquistFreq[j]) * slope;
				}
				else if (idx == mtfout.cols - 1) {
					double slope = (ptr_mtf[idx] - ptr_mtf[idx - 1]) / (ptr_freq[idx] - ptr_freq[idx - 1]);
					tt = ptr_mtf[idx] - (ptr_freq[idx] - NyquistFreq[j]) * slope;
				}
				NyquistFreqMTF.push_back((tt < 1 ? tt : 0.98));
				j++;
			}
		}
	}
	else if (mode == 1) {
		// 最邻近插值
		for (int idx0 = 0; idx0 < NyquistFreq.size(); ++idx0) {
			int idx1 = 1;// 0 不参与对比
			for (; idx1 < mtfout.cols; ++idx1) {
				if (ptr_freq[idx1] > NyquistFreq[idx0]) break;
			}
			if (abs(ptr_freq[idx1] - NyquistFreq[idx0]) > abs(ptr_freq[idx1 - 1] - NyquistFreq[idx0])) {
				idx1 -= 1;
			}
			min_idx[idx0] = idx1;
		}

		for (int i = 0; i < mtfout.rows - 1; ++i) {
			double* ptr_mtf = mtf_tmp.ptr<double>(i + 1);
			for (int idx : min_idx) {
				double tt = ptr_mtf[idx];
				NyquistFreqMTF.push_back((tt < 1 ? tt : 0.98));
			}
		}
	}
	return NyquistFreqMTF;
}

int sfrmat3::calcOneRoiStart(std::vector<double>& MTFrdPercentFreq, std::vector<double>& NyquistFreqMTF, std::vector<double>& fitCoeff, cv::Mat& mtfout,
	const cv::Mat& roiImg, const std::vector<double>& mtfPeakrdPercent, const std::vector<double>& NyquistFreq)
{
	// mtfout --[freq,mtf]
	int ret = calcSingleChannelMTF(mtfout, fitCoeff, roiImg);
	MTFrdPercentFreq = calcMTFrdPercentFreq(mtfout, fitCoeff, mtfPeakrdPercent);
	NyquistFreqMTF = calcNyquistFreqMTF(mtfout, NyquistFreq, 2);
	return ret;
}

int sfrmat3::calcAllRoiStart(const cv::Mat& srcImg, const std::vector<std::vector<std::vector<int> > >& rois,
	const std::vector<double>& mtfPeakrdPercent, const std::vector<double>& NyquistFreq)
{
	// mtfout     [[freq,lum_mtf],...]         (2 * N) or  [[ freq,lum_mtf,B_mtf, G_mtf,R_mtf],...]                                      (4 * N)
	// fitCoeff   [[ lum_k,lum_b],...]         (2 * N) or  [[ lum_k,lum_b,B_k,B_b,G_k,G_b,R_k,R_b],...]                                  (8 * N)
	// mtf3050    [[ lum_mtf30,lum_mtf50],...] (2 * N) or  [[ lum_mtf30,lum_mtf50,B_mtf30,B_mtf50,G_mtf30,G_mtf50,R_mtf30,R_mtf50],...]  (8 * N)

	//------ 初始化参数 Initialization parameters
	mtfValueList.clear();
	fitCoeffList.clear();
	MTFrdPercentFreqList.clear();
	NyquistFreqMTFList.clear();

	cv::Mat lumImg;
	if (this->imgSrc.channels() > 1) {
		std::vector<cv::Mat> rgb3Channels;
		cv::split(imgSrc, rgb3Channels);
		lumImg = weight[2] * rgb3Channels[0] + weight[1] * rgb3Channels[1] + weight[0] * rgb3Channels[2];
	}
	else {
		lumImg = srcImg.clone();
	}

	cv::Mat mtfout, mtf3050;
	int ret = 0;
	std::vector<double> fitCoeff;
//#pragma omp parallel  for
	for (int i = 0; i < rois.size(); ++i) {
		cv::Mat tmpImg = lumImg(cv::Rect(rois[i][0][0], rois[i][0][1], rois[i][1][0], rois[i][1][1]));
		//std::cout << i<<": " << cv::Rect(rois[i][0][0], rois[i][0][1], rois[i][1][0], rois[i][1][1]) << std::endl;

		ret = calcSingleChannelMTF(mtfout, fitCoeff, tmpImg);
		std::vector<double> MTFrdPercentFreq = calcMTFrdPercentFreq( mtfout, fitCoeff, mtfPeakrdPercent);
		std::vector<double> NyquistFreqMTF = calcNyquistFreqMTF( mtfout, NyquistFreq, 2);

		this->mtfValueList.emplace_back(mtfout);
		this->fitCoeffList.emplace_back(fitCoeff);
		this->MTFrdPercentFreqList.emplace_back(MTFrdPercentFreq);
		this->NyquistFreqMTFList.emplace_back(NyquistFreqMTF);
	}
	return 0;
}

py::tuple sfrmat3::calcOneRoiStart_API(py::array_t<uint8_t>& py_roiImg, std::vector<double>& mtfPeakrdPercent, std::vector<double>& NyquistFreq)
{
	py::buffer_info buf = py_roiImg.request();                                      // 请求获取 py::array_t<uint8_t>对象 py_roiImage的缓冲区信息(buffer_info可以在不进行复制的情况下数据共享)
	cv::Mat roiImg(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);   // 创建一个openCV的Mat对象, 用于存储图像数据
	std::vector<double> MTFrdPercentFreq, NyquistFreqMTF, fitCoeff;                 // 定义一些用于储存计算结果的向量
	cv::Mat mtfout;
	this->calcOneRoiStart(MTFrdPercentFreq, NyquistFreqMTF, fitCoeff, mtfout, roiImg, mtfPeakrdPercent, NyquistFreq);  

	py::array_t<double> mtfdata = py::array_t<double>({ mtfout.rows,mtfout.cols }, (double*)mtfout.data);  // 将计算结果mtfout转换为py::array_t<double>对象
	return py::make_tuple(mtfdata, MTFrdPercentFreq, NyquistFreqMTF, fitCoeff);
}

py::tuple sfrmat3::calcAllRoiStart_API(py::array_t<uint8_t>& py_srcImg, std::vector<std::vector<std::vector<int> > >& rois, std::vector<double>& mtfPeakrdPercent, std::vector<double>& NyquistFreq)
{
	std::cout << "---calculate SFR of source image start.(infos: " << rois.size() << " " << mtfPeakrdPercent.size() << " " << NyquistFreq.size() << ")" << std::endl;
	clock_t _start = clock();

	py::buffer_info buf = py_srcImg.request();
	cv::Mat srcImg(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
	this->calcAllRoiStart(srcImg, rois, mtfPeakrdPercent, NyquistFreq);
	std::vector< py::array_t <double> > mtfdataList;
	for (auto& cc : this->mtfValueList) {
		mtfdataList.push_back(py::array_t<double>({ cc.rows,cc.cols }, (double*)cc.data));
	}
	clock_t _end = clock();
	std::cout << "-----calculate SFR of source image end. (calc time : " << (double)(_end - _start) << " ms )" << std::endl;
	return py::make_tuple(mtfdataList, this->MTFrdPercentFreqList, this->NyquistFreqMTFList,this->fitCoeffList);
}