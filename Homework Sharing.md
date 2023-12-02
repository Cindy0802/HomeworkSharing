# Homework Sharing

# 1.作业一：并行矩阵乘法

## 项目简介：

编写⼀个基于oneAPI的C++/SYCL程序来执行矩阵乘法操作。需要考虑大尺寸矩阵的乘法操作以及不同线程之间的数据依赖关系。通常在实现矩阵乘法时，可以使用块矩阵乘法以及共享内存来提高计算效率。

## **主要思路：**

1. **Kernel函数设计：** **`gpu_kernel`** 函数是 SYCL 核函数，用于执行矩阵乘法的并行计算。核函数使用二维的全局和本地工作组范围，并在每个工作项内进行矩阵乘法的计算。使用了分块矩阵乘法的思想，将矩阵分为小块，每个工作项计算其中的一小块。
2. **矩阵分块：** 将输入矩阵 A、B、C 分块，通过循环迭代计算每个小块的矩阵乘法，最终得到输出矩阵 C。
3. **性能计时：** 使用 SYCL 提供的事件机制来测量 GPU 核函数的执行时间，最后计算平均时间。
4. **性能指标输出：** 输出计算的问题规模、工作组大小、性能（Flops）以及 GPU 计算时间等信息。

## **优化手段：**

1. **并行计算：** 通过使用 SYCL 提供的并行计算能力，核函数内的计算被分配到不同的工作项上，从而实现并行计算。
2. **矩阵分块优化：** 采用分块矩阵乘法的思想，减小了每个工作项需要处理的数据量，有利于提高数据缓存的命中率，减少数据传输时间。

将矩阵分为大小为 **`tileY x tileX`** 的小块，**`tileY`** 和 **`tileX`** 分别表示矩阵乘法中每个工作项计算的小块的行数和列数。每个工作项负责计算其中的一小块，这种矩阵分块的方式能更好地利用局部数据的缓存，减少全局内存的访问次数，从而提高计算效率。

尝试了多种tileX和tileY的组合，并记录计算时间：

![Untitled](Homework%20Sharing%203a1b261b98ed4dd19f02978a178c462d/Untitled.png)

最终选用tileX=8，tileY=8时，所有时间最少，约为0.522ms。

## 执行结果：

两个512*512的矩阵相乘，**GPU计算时间为0.522808ms**

![Untitled](Homework%20Sharing%203a1b261b98ed4dd19f02978a178c462d/Untitled%201.png)

## 代码：

```cpp
#include <iostream>
#include <CL/sycl.hpp>

#define random_float() (rand() / double(RAND_MAX))

using namespace std;
using namespace sycl;

#define tileY 8
#define tileX 8

//返回执行时间
double gpu_kernel(float *A, float *B, float *C, 
                  int M, int N, int K, 
                  int BLOCK, sycl::queue &q) {

    auto grid_rows = M / tileY;
    auto grid_cols = N / tileX;
    auto local_ndrange  = range<2>(BLOCK, BLOCK);
    auto global_ndrange = range<2>(grid_rows, grid_cols);

    double duration = 0.0f;

    //核函数调用
    auto e = q.submit([&](sycl::handler &h) {
        h.parallel_for<class k_name_t>(
            sycl::nd_range<2>(global_ndrange, local_ndrange), [=](sycl::nd_item<2> index) {

                int row = tileY * index.get_global_id(0);
                int col = tileX * index.get_global_id(1);

                float sum[tileY][tileX] = {0.0f};
                float subA[tileY];
                float subB[tileX];

                //并行计算
                for (int m = 0; m < tileY; m++) {
                    subA[m] = A[(row + m) * N];
                }

                for (int k = 0; k < N; k++) {
                    
                    for (int p = 0; p < tileX; p++) {
                        subB[p] = B[k * N + p + col];
                    }

                    for (int m = 0; m < tileY; m++) {
                        for (int p = 0; p < tileX; p++) {
                            sum[m][p] += subA[m] * subB[p];
                        }
                    }
                }

                
                for (int m = 0; m < tileY; m++) {
                    for (int p = 0; p < tileX; p++) {
                        C[(row + m) * N + col + p] = sum[m][p];
                    }
                }

            });
    });
    e.wait();

    duration += (e.get_profiling_info<info::event_profiling::command_end>() -
    e.get_profiling_info<info::event_profiling::command_start>()) /1000.0f/1000.0f;

    return(duration);
}

//矩阵乘法
int gemm(const int M, 
         const int N, 
         const int K, 
         const int block_size,
         const int iterations, 
         sycl::queue &q) {

    cout << "问题规模: c(" << M << "," <<  N << ") ="
         << " a(" << M << "," << K << ") *" 
         << " b(" << K << "," << N << ")\n";

    //分配内存
    auto A = malloc_shared<float>(M * K, q);
    auto B = malloc_shared<float>(K * N, q);
    auto C = malloc_shared<float>(M * N, q);

    for(int i=0; i < M * K; i++) {
        A[i] = random_float();
    }

    for(int i=0; i < K * N; i++) {
        B[i] = random_float();
    }

    for(int i=0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    double flopsPerMatrixMul
        = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    double duration_gpu = 0.0f;

    int warmup = 10;
    for (int run = 0; run < iterations + warmup; run++) {
        float duration = gpu_kernel(A, B, C, M, N, K, block_size, q);
        if(run >= warmup) duration_gpu += duration;
    }
    duration_gpu = duration_gpu / iterations;

    printf("\nGEMM 大小 M = %d, N = %d, K = %d", M, N, K);
    printf("\n工作组大小 = %d * %d, tile_X = %d, tile_Y = %d", block_size, block_size, tileX, tileY);
    printf("\n性能 Flops = %lf, \n" 
            "GPU 计算时间 = %lf (ms); \n",
            flopsPerMatrixMul, duration_gpu);

    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}

int main() {
    auto propList = cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()};
    queue my_gpu_queue( cl::sycl::gpu_selector{} , propList);

    int errCode = gemm(512, 512, 512, // GEMM 大小, M, N, K
                        4,             // 工作组大小
                        10,            // 重复次数 
                        my_gpu_queue);

    return errCode;
}
```

# 2.作业二：并行排序算法

## 项目简介：

使用基于oneAPI的C++/SYCL实现⼀个高效的并行归并排序。需要考虑数据的分割和合并以及线程之间的协作。

## 执行结果：

![Untitled](Homework%20Sharing%203a1b261b98ed4dd19f02978a178c462d/Untitled%202.png)

## 代码：

```cpp
%%writefile lab/gpu_sample.cpp
//作业二：并行排序算法
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

// 归并排序合并操作
void merge(vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // 创建临时数组 L 和 R
    vector<int> L(n1), R(n2);

    // 复制数据到临时数组 L 和 R
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[middle + 1 + j];

    // 合并临时数组 L 和 R 到 arr
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // 复制 L 的剩余元素（如果有的话）
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // 复制 R 的剩余元素（如果有的话）
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// 归并排序递归函数
void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        // 找到数组的中间点
        int middle = left + (right - left) / 2;

        // 递归地排序左半部分和右半部分
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);

        // 合并已排序的两部分
        merge(arr, left, middle, right);
    }
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};

    cout << "原始数组: ";
    for (int i : arr) {
        cout << i << " ";
    }
    cout << endl;

    // 计算排序时间
    auto start_time = chrono::high_resolution_clock::now();
    mergeSort(arr, 0, arr.size() - 1);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;

    cout << "排序后的数组: ";
    for (int i : arr) {
        cout << i << " ";
    }
    cout << endl;

    cout << "计算时间: " << duration.count() << " 毫秒" << endl;

    return 0;
}
```

# 3.作业三：图像卷积并行加速

## 项目简介：

使用基于oneAPI的C++/SYCL实现一个用于计算图像的卷积操作。输⼊为一个图像矩阵和一个卷积核矩阵，输出为卷积后的图像。

## 执行结果：

![Untitled](Homework%20Sharing%203a1b261b98ed4dd19f02978a178c462d/Untitled%203.png)

### 代码：

```cpp
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
```

# 4.收获

学习Intel OneAPI让我深入了解异构计算和跨架构开发的重要性。通过使用DPC++编程语言，我能够更高效地利用Intel的各种硬件加速器，如CPU、GPU和FPGA。OneAPI的统一编程模型使得代码跨不同硬件平台可移植，提高了开发效率。同时，OneAPI提供了丰富的工具集，包括性能分析工具和调试器，帮助我更好地优化和调试程序。总体而言，通过学习Intel OneAPI，我不仅提升了对异构计算的理解，还获得了更广泛的跨平台开发经验。