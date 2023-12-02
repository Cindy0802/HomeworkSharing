%%writefile lab/gpu_sample.cpp
//作业一：并行矩阵乘法
#include <iostream>
#include <CL/sycl.hpp>

#define random_float() (rand() / double(RAND_MAX))

using namespace std;
using namespace sycl;

#define tileY 8
#define tileX 8

// 返回执行时间
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

// 矩阵乘法的函数
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

    int errCode = gemm(512, 512, 512, /* GEMM 大小, M, N, K */
                        4,             /* 工作组大小 */ 
                        10,            /* 重复次数 */   
                        my_gpu_queue);

    return errCode;
}

