%%writefile lab/gpu_sample.cpp
//��ҵ����ͼ�������м���
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

// �����ά��������
typedef vector<vector<int>> Matrix;

// ��ӡ����
void printMatrix(const Matrix& matrix) {
    for (const auto& row : matrix) {
        for (int value : row) {
            cout << value << " ";
        }
        cout << endl;
    }
}

// �Զ�ά������о��
Matrix convolution(const Matrix& input, const Matrix& kernel) {
    int inputRows = input.size();
    int inputCols = input[0].size();
    int kernelRows = kernel.size();
    int kernelCols = kernel[0].size();

    // ��������ľ����С
    int resultRows = inputRows - kernelRows + 1;
    int resultCols = inputCols - kernelCols + 1;

    // ��ʼ������������
    Matrix result(resultRows, vector<int>(resultCols, 0));

    // ִ�о������
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

    // ��¼��ʼʱ��
    auto start = high_resolution_clock::now();

    // ִ�о��
    Matrix result = convolution(inputImage, kernel);

    // ��¼����ʱ��
    auto stop = high_resolution_clock::now();

    // ����ִ��ʱ��
    auto duration = duration_cast<microseconds>(stop - start);

    // ������
    cout << "����ͼ��" << endl;
    printMatrix(inputImage);

    cout << "\n����ˣ�" << endl;
    printMatrix(kernel);

    cout << "\n��������" << endl;
    printMatrix(result);

    cout << "\nִ��ʱ��: " << duration.count() << " ΢��" << endl;

    return 0;
}

