#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <algorithm>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "DanglingPointer"

void printArr(int N, int M, const float *arr) {
    for (int i = 0; i < N; ++i) {
        const float *ptr = arr + i * M;
        for (int j = 0; j < M; ++j) {
            std::cout << ptr[j] << " ";
        }
        std::cout << "\n";
    }
}

void initArray(float *arr, int size, std::string filename = "in.txt") {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "error occurred" << std::endl;
        return;
    }
    for (int i = 0; i < size && !in.eof(); ++i) {
        float tmp;
        in >> tmp;
        arr[i] = tmp;
    }
}
namespace auto_vectorizing{
    void multMatrices(int N, const float *A, const float *B, float *R) {
        for (int i = 0; i < N; ++i) {
            float *r = R + i * N;
            for (int j = 0; j < N; ++j) {
                const float a = A[i * N + j];
                const float *b = B + j * N;
                for (int k = 0; k < N; ++k) {
                    r[k] += a * b[k];
                }
            }
        }
    }

    inline float rowSum(int N, const float *a) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += std::abs(a[i]);
        }
        return sum;
    }

    float findMaxRowSum(int N, const float *a) {
        float max = 0.0f;
        for (int i = 0; i < N; ++i) {
            const float *ptr = a + i * N;
            float tmpSum = rowSum(N, ptr);
            max = std::max(max, tmpSum);
        }
        return max;
    }

    float findMaxColumnSum(int N, const float *a) {
        float *buf = new float[N];
        for (int i = 0; i < N; ++i) {
            buf[i] = 0;
        }

        for (int i = 0; i < N; ++i) {
            const float *ptr = a + i * N;
            for (int j = 0; j < N; ++j) {
                buf[j] += std::abs(ptr[j]);
            }
        }
        float max = 0.0f;
        for (int i = 0; i < N; ++i) {
            max = std::max(max, buf[i]);
        }

        delete[] buf;
        return max;
    }

    void calcB(const float *a, float *res, int N) {
        float k = 1.0f / (findMaxRowSum(N, a) * findMaxColumnSum(N, a));
        for (int n = 0; n < N * N; n++) {
            int i = n / N;
            int j = n % N;
            res[n] = a[N * j + i] * k;
        }
    }

    void calcR(const float *a, const float *b, float *r, int N) {
        multMatrices(N, b, a, r);
        for (int i = 0; i < N; ++i) {
            float *rr = r + i * N;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    rr[j] = -rr[j];
                } else {
                    rr[j] = 1 - rr[j];
                }
            }
        }
    }

    void addInFirst(float *a, const float *b, int size) {
        for (int i = 0; i < size; ++i) {
            a[i] += b[i];
        }
    }

    inline float *getI(int N) {
        float *ret = new float[N * N];
        for (int i = 0; i < N; ++i) {
            float *p = ret + i * N;
            for (int j = 0; j < N; ++j) {
                if (i != j) {
                    p[j] = 0;
                } else {
                    p[j] = 1;
                }
            }
        }
        return ret;
    }

    inline void fillWithZeros(float *a, int size) {
        for (int i = 0; i < size; ++i) {
            a[i] = 0;
        }
    }

    inline float *getArrayWithZeros(int size) {
        float *ret = new float[size];
        fillWithZeros(ret, size);
        return ret;
    }

    void calculateSumRow(float *sum, const float *r, float *rAcc, int N, int iterations) {
        float *buf = getArrayWithZeros(N * N);
        addInFirst(sum, rAcc, N * N);
        multMatrices(N, r, rAcc, buf);
        std::copy(buf, buf + N * N, rAcc);
        for (int i = 1; i < iterations; ++i) {
            fillWithZeros(buf, N * N);
            addInFirst(sum, rAcc, N * N);
            multMatrices(N, r, rAcc, buf);
            std::copy(buf, buf + N * N, rAcc);
        }
        delete[] buf;
    }
}

int main() {
    //initialization
    const int N = 4;

    int size = N * N;
    float *a = new float[size];
    initArray(a, size, "in_2048_2048.txt");

    float *b = auto_vectorizing::getArrayWithZeros(size);
    float *r = auto_vectorizing::getArrayWithZeros(size);
    float *rAcc = new float[size];

    float *sum = auto_vectorizing::getI(N);
    float *reverse = auto_vectorizing::getArrayWithZeros(size);

    auto start = std::chrono::steady_clock::now();

    auto_vectorizing::calcB(a, b, N);
    auto_vectorizing::calcR(a, b, r, N);
    std::copy(r, r + size, rAcc);
    auto_vectorizing::calculateSumRow(sum, r, rAcc, N, 10000);
    auto_vectorizing::multMatrices(N, sum, b, reverse);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    delete[] a;
    delete[] b;
    delete[] r;
    delete[] rAcc;
    delete[] sum;
    delete[] reverse;

    return 0;
}

#pragma clang diagnostic pop