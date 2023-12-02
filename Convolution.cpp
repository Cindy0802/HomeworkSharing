%%writefile lab/gpu_sample.cpp
//作业三：图像卷积并行加速
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// 定义二维向量类型
typedef vector<vector<int>> Matrix;

// 打印矩阵
void printMatrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (int value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

// 对二维矩阵进行卷积
Matrix convolution(const Matrix& input, const Matrix& kernel) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    // 计算卷积后的矩阵大小
    int resultRows = inputRows - kernelRows + 1;
    int resultCols = inputCols - kernelCols + 1;

    // 初始化卷积结果矩阵
    Matrix result(resultRows, vector<int>(resultCols, 0));

    // 执行卷积操作
    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultCols; ++j) {
            for (int k = 0; k < kernelRows; ++k) {
                for (int l = 0; l < kernelCols; ++l) {
                    result[i][j] += input[i + k][j + l] * kernel[k][l];
                }
            }
        }
    }

    return result;
}

int main() {

    Matrix inputImage = {
        { 5, 12, 25, 41, 57, 72, 85, 94, 100, 82 },
    { 17, 23, 38, 49, 63, 79, 92, 10, 96, 41 },
    { 31, 45, 50, 66, 78, 12, 27, 37, 58, 91 },
    { 43, 58, 72, 89, 95, 9, 21, 34, 47, 69 },
    { 57, 63, 78, 10, 22, 37, 49, 52, 61, 79 },
    { 63, 75, 82, 94, 1, 14, 27, 38, 43, 55 },
    { 72, 86, 93, 8, 23, 39, 47, 57, 68, 78 },
    { 85, 96, 9, 16, 28, 38, 46, 59, 67, 76 },
    { 92, 4, 18, 29, 35, 49, 51, 62, 74, 87 },
    { 8, 21, 35, 42, 54, 69, 75, 82, 97, 100 }
    };


    Matrix kernel = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    // 记录开始时间
    auto start = high_resolution_clock::now();

    // 执行卷积
    Matrix result = convolution(inputImage, kernel);

    // 记录结束时间
    auto stop = high_resolution_clock::now();

    // 计算执行时间
    auto duration = duration_cast<microseconds>(stop - start);

    // 输出结果
    cout << "输入图像：" << endl;
    printMatrix(inputImage);

    cout << "\n卷积核：" << endl;
    printMatrix(kernel);

    cout << "\n卷积结果：" << endl;
    printMatrix(result);

    cout << "\n执行时间: " << duration.count() << " 微秒" << endl;

    return 0;
}

