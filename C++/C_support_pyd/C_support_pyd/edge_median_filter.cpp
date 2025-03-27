#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

// 示例函数：对输入二维 NumPy 数组进行边缘中值滤波（使用 OpenMP 并行加速）
py::array_t<double> edge_median_filter(py::array_t<double> image, py::array_t<int> offsets,int pad) 
{
    py::buffer_info buf_img = image.request();
    py::buffer_info buf_off = offsets.request();

    if (buf_img.ndim != 2)
        throw std::runtime_error("Image must be 2-dimensional");
    if (buf_off.ndim != 1)
        throw std::runtime_error("Offsets must be 1-dimensional");

    int rows = buf_img.shape[0];
    int cols = buf_img.shape[1];
    int num_edges = buf_off.shape[0];
    int mid = num_edges / 2;

    double* ptr_img = static_cast<double*>(buf_img.ptr);
    int* ptr_off = static_cast<int*>(buf_off.ptr);

    // 创建一个与输入同形状的输出数组
    auto result = py::array_t<double>(buf_img.shape);
    py::buffer_info buf_out = result.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    // 并行处理内区域
#pragma omp parallel for collapse(2)
    for (int i = pad; i < rows - pad; i++) {
        for (int j = pad; j < cols - pad; j++) {
            int top_left_index = (i - pad) * cols + (j - pad);
            int window_rows = 2 * pad + 1;
            int window_cols = 2 * pad + 1;
            std::vector<double> edge_values(num_edges);
            for (int k = 0; k < num_edges; k++) {
                int off = ptr_off[k];
                int local_row = off / window_cols;
                int local_col = off % window_cols;
                int image_index = top_left_index + local_row * cols + local_col;
                edge_values[k] = ptr_img[image_index];
            }
            std::nth_element(edge_values.begin(), edge_values.begin() + mid, edge_values.end());
            double median = edge_values[mid];
            ptr_out[i * cols + j] = median;
        }
    }

    // 对边缘区域直接复制原始像素值
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i < pad || i >= rows - pad || j < pad || j >= cols - pad) {
                ptr_out[i * cols + j] = ptr_img[i * cols + j];
            }
        }
    }

    return result;
}

PYBIND11_MODULE(edge_median_filter, m) {
    m.doc() = "Median filter module implemented in C++ with pybind11 and OpenMP";
    m.def("edge_median_filter", &edge_median_filter,
        "edge_median_filter",
        py::arg("image"), py::arg("offsets"), py::arg("pad"));
}
