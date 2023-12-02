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

