#include "sfrmat3.h"


PYBIND11_MODULE(MTF3_CSupport, m)
{
	m.doc() = "sfrmat3 API";  // ģ���ĵ��ַ���

	py::class_<sfrmat3>(m, "sfrmat3")                                      // ��C++��sfrmat3�󶨵�pythonģ����,������Ϊsfrmat3
		.def(py::init<float, double, std::vector<double>, std::string>())  // ����Ĺ��캯��,�����ĸ�����; float, double, sfd::vector<double>. std::string
		.def("SetParams", &sfrmat3::SetParams, "Initialization parameter") // ��SetParams����, ���ṩ�ĵ��ַ���
		.def("calcOneRoiStart_API", &sfrmat3::calcOneRoiStart_API, "calc one roi sfr value Start API flow")  // ��calcOneRoiStart_API����,���ṩ�ĵ��ַ���
		.def("calcAllRoiStart_API", &sfrmat3::calcAllRoiStart_API, "calc one hole image`s all roi sfr value Start API flow")  // ��calcAllRoiStart_API����,���ṩ�ĵ��ַ���
		.def("versionInfo", &sfrmat3::versionInfo, "sfr version");         // ��versionInfo����, ���ṩ�ĵ��ַ���
}

void testROIImage() {
	cv::Mat img = cv::imread("D:/tem/Temp/SFR/fail_roi.png", 0);
	clock_t _start = clock();
	std::vector<double> weight = std::vector<double>({ 0.213, 0.715, 0.072 });
	std::vector<double> mtfPeakrdPercent({0.5, 0.2 });
	std::vector<double> NyquistFreq({ 0.25,0.125 });
	std::vector<double> MTFrdPercentFreq, NyquistFreqMTF, fitCoeff;
	cv::Mat mtfout;

	sfrmat3 sfr(1.0, 1.0, weight, "none");
	int ret = sfr.calcOneRoiStart(MTFrdPercentFreq, NyquistFreqMTF, fitCoeff, mtfout, img, mtfPeakrdPercent, NyquistFreq);
	clock_t _end = clock();
	std::cout << "  calc time : " << (double)(_end - _start) << " ms calc result : " << ret << std::endl;
	std::cout << "  MTFrdPercentFreq : [";
	for (int i = 0; i < MTFrdPercentFreq.size(); ++i) {
		std::cout << MTFrdPercentFreq[i] << ", ";
	}
	std::cout <<"]" << std::endl;

	std::cout << "  NyquistFreqMTF : [";
	for (int i = 0; i < NyquistFreqMTF.size(); ++i) {
		std::cout << NyquistFreqMTF[i] << ", ";
	}
	std::cout << "]" << std::endl;

	std::cout << "  fitCoeff : [";
	for (int i = 0; i < fitCoeff.size(); ++i) {
		std::cout << fitCoeff[i] << ", ";
	}
	std::cout << "]" << std::endl;
	std::cout <<" \n  MTF : " << mtfout.t() << std::endl;
}

void testSRCImage()
{
	cv::Mat img = cv::imread("./aSRC.png", 0);
	std::vector<std::vector<std::vector<int> > > rois;
	clock_t _start = clock();
	std::vector<double> weight = std::vector<double>({ 0.213, 0.715, 0.072 });
	std::vector<double> mtfPeakrdPercent({ 0.5, 0.2 });
	std::vector<double> NyquistFreq({ 0.25,0.125 });
	std::vector<double> MTFrdPercentFreq, NyquistFreqMTF, fitCoeff;

	sfrmat3 sfr(1.0, 1.0, weight, "none");
	int ret = 0;
	ret = sfr.calcAllRoiStart(img, rois, mtfPeakrdPercent, NyquistFreq);

	clock_t _end = clock();
	std::cout << "  calc time : " << (double)(_end - _start) << " ms calc result : " << ret << std::endl;
}

int main(int argc, char** argv)
{
	testROIImage();
	//testSRCImage();
	return 0;
}