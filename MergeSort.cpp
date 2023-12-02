%%writefile lab/gpu_sample.cpp
//��ҵ�������������㷨
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

// �鲢����ϲ�����
void merge(vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // ������ʱ���� L �� R
    vector<int> L(n1), R(n2);

    // �������ݵ���ʱ���� L �� R
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[middle + 1 + j];

    // �ϲ���ʱ���� L �� R �� arr
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

    // ���� L ��ʣ��Ԫ�أ�����еĻ���
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // ���� R ��ʣ��Ԫ�أ�����еĻ���
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// �鲢����ݹ麯��
void mergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        // �ҵ�������м��
        int middle = left + (right - left) / 2;

        // �ݹ��������벿�ֺ��Ұ벿��
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);

        // �ϲ��������������
        merge(arr, left, middle, right);
    }
}

int main() {
    vector<int> arr = {12, 11, 13, 5, 6, 7};

    cout << "ԭʼ����: ";
    for (int i : arr) {
        cout << i << " ";
    }
    cout << endl;

    // ��������ʱ��
    auto start_time = chrono::high_resolution_clock::now();
    mergeSort(arr, 0, arr.size() - 1);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end_time - start_time;

    cout << "����������: ";
    for (int i : arr) {
        cout << i << " ";
    }
    cout << endl;

    cout << "����ʱ��: " << duration.count() << " ����" << endl;

    return 0;
}

