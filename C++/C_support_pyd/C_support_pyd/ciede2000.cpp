#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <tuple>

namespace py = pybind11;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
std::tuple<double, double> calcu_ciede2000(const std::vector<double>& Lab_1, const std::vector<double>& Lab_2) {
    // 预计算常量
    const double C_25_7 = 6103515625.0;  // 25**7
    const double two_pi = 2.0 * M_PI;
    const double pi_6 = M_PI / 6.0;
    const double pi_30 = M_PI / 30.0;
    const double deg63 = 63.0 * M_PI / 180.0;

    // 提取 Lab 颜色值
    double L1 = Lab_1[0], a1 = Lab_1[1], b1 = Lab_1[2];
    double L2 = Lab_2[0], a2 = Lab_2[1], b2 = Lab_2[2];

    // 初始色度计算
    double C1 = std::sqrt(a1 * a1 + b1 * b1);
    double C2 = std::sqrt(a2 * a2 + b2 * b2);
    double C_ave_init = (C1 + C2) / 2.0;

    // 计算 G 因子
    double C_ave_init_pow7 = std::pow(C_ave_init, 7);
    double G = 0.5 * (1 - std::sqrt(C_ave_init_pow7 / (C_ave_init_pow7 + C_25_7)));

    // 调整 a 值
    double a1_ = (1 + G) * a1;
    double a2_ = (1 + G) * a2;
    double b1_ = b1, b2_ = b2;

    // 计算新的色度
    double C1_ = std::sqrt(a1_ * a1_ + b1_ * b1_);
    double C2_ = std::sqrt(a2_ * a2_ + b2_ * b2_);

    // 计算色相角 h
    double h1_ = (a1_ == 0.0 && b1_ == 0.0) ? 0.0 : std::atan2(b1_, a1_);
    double h2_ = (a2_ == 0.0 && b2_ == 0.0) ? 0.0 : std::atan2(b2_, a2_);

    if (h1_ < 0) h1_ += two_pi;
    if (h2_ < 0) h2_ += two_pi;

    // 计算亮度差、色度差
    double dL_ = L2 - L1;
    double dC_ = C2_ - C1_;

    // 计算色相差
    double dh_ = h2_ - h1_;
    if (C1_ * C2_ != 0.0) {
        if (dh_ > M_PI) dh_ -= two_pi;
        else if (dh_ < -M_PI) dh_ += two_pi;
    }
    else {
        dh_ = 0.0;
    }

    double dH_ = 2.0 * std::sqrt(C1_ * C2_) * std::sin(dh_ / 2.0);

    // 计算平均亮度、色度
    double L_ave = (L1 + L2) / 2.0;
    double C_ave_final = (C1_ + C2_) / 2.0;

    // 计算平均色相
    double h_ave;
    double dh_abs = std::abs(h1_ - h2_);
    double sh_sum = h1_ + h2_;

    if (C1_ * C2_ == 0.0) {
        h_ave = h1_ + h2_;
    }
    else {
        if (dh_abs <= M_PI) {
            h_ave = (h1_ + h2_) / 2.0;
        }
        else {
            if (sh_sum < two_pi) h_ave = (h1_ + h2_) / 2.0 + M_PI;
            else h_ave = (h1_ + h2_) / 2.0 - M_PI;
        }
    }

    // 计算 T 因子
    double T = 1 - 0.17 * std::cos(h_ave - pi_6)
        + 0.24 * std::cos(2.0 * h_ave)
        + 0.32 * std::cos(3.0 * h_ave + pi_30)
        - 0.2 * std::cos(4.0 * h_ave - deg63);

    // 计算 dTheta
    double h_ave_deg = h_ave * 180.0 / M_PI;
    if (h_ave_deg < 0.0) h_ave_deg += 360.0;
    if (h_ave_deg > 360.0) h_ave_deg -= 360.0;

    double dTheta = 30.0 * std::exp(-std::pow((h_ave_deg - 275.0) / 25.0, 2));

    // 计算 R_C
    double C_ave_final_pow7 = std::pow(C_ave_final, 7);
    double R_C = 2.0 * std::sqrt(C_ave_final_pow7 / (C_ave_final_pow7 + C_25_7));

    double S_C = 1 + 0.045 * C_ave_final;
    double S_H = 1 + 0.015 * C_ave_final * T;
    double S_L = 1 + 0.015 * std::pow(L_ave - 50.0, 2) / std::sqrt(20.0 + std::pow(L_ave - 50.0, 2));

    double R_T = -std::sin(dTheta * M_PI / 90.0) * R_C;

    // 计算最终 dE
    double f_L = dL_ / S_L;
    double f_C = dC_ / S_C;
    double f_H = dH_ / S_H;

    double dE_00 = std::sqrt(f_L * f_L + f_C * f_C + f_H * f_H + R_T * f_C * f_H);
    double dC_00 = std::sqrt(f_C * f_C + f_H * f_H + R_T * f_C * f_H);

    return { dE_00, dC_00 };
}

// 绑定到 Python
PYBIND11_MODULE(ciede2000, m) {
    m.def("calcu_ciede2000", &calcu_ciede2000, "Calculate CIEDE2000 color difference");
}
