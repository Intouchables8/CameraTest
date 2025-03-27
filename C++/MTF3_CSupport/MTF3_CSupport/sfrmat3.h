#pragma once
#include <ctime>
#include <opencv2/opencv.hpp>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define EPS 1e-6

class sfrmat3
{
public:
    sfrmat3(float _gamma, double _intervalUnit, std::vector<double> _weight, std::string _oecfname);
    // 只能计算单通道图片，若是 3 通道 RGB 则强制转单通道
    // 所有 cv::Mat 存储的数据都改为了 行优先（图片数据不变）

public:
    // input param
    void SetParams(float _gamma = 1.0, double _intervalUnit = 1.0, std::vector<double> _weight = std::vector<double>({ 0.213, 0.715, 0.072 }), std::string _oecfname = "none");
    // calc flow
    int calcOneRoiStart(std::vector<double>& MTFrdPercentFreq, std::vector<double>& NyquistFreqMTF, std::vector<double>& fitCoeff,cv::Mat& mtfout,
        const cv::Mat& roiImg, const std::vector<double>& mtfPeakrdPercent, const std::vector<double>& NyquistFreq);

    int calcAllRoiStart(const cv::Mat& srcImg, const std::vector<std::vector<std::vector<int> > >& rois, 
        const std::vector<double>& mtfPeakrdPercent, const std::vector<double>& NyquistFreq);

    py::tuple calcOneRoiStart_API(py::array_t<uint8_t>& py_roiImg,std::vector<double>& mtfPeakrdPercent, std::vector<double>& NyquistFreq);
    py::tuple calcAllRoiStart_API(py::array_t<uint8_t>& py_srcImg, std::vector<std::vector<std::vector<int> > >& rois, std::vector<double>& mtfPeakrdPercent, std::vector<double>& NyquistFreq);

    // output version
    std::string versionInfo();
private:
    int calcSingleChannelMTF(cv::Mat& mtfout, std::vector<double>& fitCoeff, const cv::Mat& srcimg);
    std::vector<double> calcMTFrdPercentFreq(const cv::Mat& mtfout, const std::vector<double>& fitCoeff, std::vector<double> mtfPeakrdPercent);
    std::vector<double> calcNyquistFreqMTF(const cv::Mat& mtfout, const std::vector<double>& NyquistFreq, int mode);

private:
    cv::Mat gammaTransform(cv::Mat& srcImage, float kFactor);
    cv::Mat matRotateCounterClockWise90(cv::Mat src);
    cv::Mat rotatev2(cv::Mat img, bool& hor2ver);
    cv::Mat ahamming(int n, double mid);
    cv::Mat derivationFun(const cv::Mat& input, cv::Mat fil);
    double centroid(const cv::Mat& input);
    std::vector<double> polyfit(std::vector<cv::Point2d>& in_point, int n);
    cv::Mat fir2fix(int n, int m);
    cv::Mat project(const cv::Mat& bb, double loc, double slope, int fac = 4);
    cv::Mat cent(const cv::Mat& a, int center);
    cv::Mat complex_abs(const cv::Mat& input);
    void sampeff(double& eff, std::vector<double>& freqval, std::vector<double>& sfrval, const cv::Mat& mtfout, const std::vector<double>& valList, double del);
    void showChart(const cv::Mat& mtfout);
private:
    // input and output
    cv::Mat imgSrc; // uint8_t
    int counts;

    float gamma;
    double intervalUnit;
    std::string oecfname;
    std::vector<double> weight;
    std::string funit;

    // tmpList freqValue and tmp mtfValue( roi number N)
    std::vector< cv::Mat > mtfValueList;
    std::vector< std::vector<double> > fitCoeffList;
    std::vector< std::vector<double> > MTFrdPercentFreqList;//[[MTF50 MTF30 ...];[MTF50 MTF30 ...]... N]
    std::vector< std::vector<double> > NyquistFreqMTFList;// [Nyquist1:[1,2,3..N] ; Nyquist2:[1,2,3...N]]
};

//extern int sfrmat3(cv::Mat& fitCoeff, cv::Mat& mtfout, cv::Mat& mtf3050, 
//	const cv::Mat& img, float gamma = 1.0, double intervalUnit = 1, std::vector<double> weight = std::vector<double>({ 0.213, 0.715, 0.072 }),std::string oecfname = "none");
//
//extern int Sharpness2MTF(std::vector<double>& vecFreq2Mtf,const cv::Mat& mtfout, std::vector<double> fixedSharpness, int mode = 1);