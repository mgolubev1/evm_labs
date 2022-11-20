#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"
#include <algorithm>
#include <cblas-openblas.h>
#include <chrono>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>

void toDefaultParams(const int N, int size, float *b, float *r, float *sum);
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
inline void makeI(int N, float *a) {
  for (int i = 0; i < N; ++i) {
    float *p = a + i * N;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        p[j] = 0;
      } else {
        p[j] = 1;
      }
    }
  }
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
namespace auto_vectorizing {
void multMatrices(int N, const float *A, const float *B, float *R) {
  for (int i = 0; i < N; ++i) {
    float *r = R + i * N;
    for (int j = 0; j < N; ++j) {
      r[j] = 0;
    }
    for (int j = 0; j < N; ++j) {
      const float a = A[i * N + j];
      const float *b = B + j * N;
      for (int k = 0; k < N; ++k) {
        r[k] += a * b[k];
      }
    }
  }
}

float findMaxRowSum(int N, const float *a) {
  float max = 0.0f;
  for (int i = 0; i < N; ++i) {
    const float *ptr = a + i * N;
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
      sum += std::abs(ptr[j]);
    }
    max = std::max(max, sum);
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
  float K = 1.0f / (findMaxRowSum(N, a) * findMaxColumnSum(N, a));
  int blocksize = 8;
  for (int i = 0; i < N; i += blocksize) {
    for (int j = 0; j < N; j += blocksize) {
      // transpose the block beginning at [i,j]
      for (int k = i; k < i + blocksize; ++k) {
        int kn = k * N;
        for (int l = j; l < j + blocksize; ++l) {
          res[k + l * N] = a[l + kn] * K;
        }
      }
    }
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

inline void addInFirst(float *a, const float *b, int size) {
  for (int i = 0; i < size; ++i) {
    a[i] += b[i];
  }
}

void calculateSumRow(float *sum, const float *r, float *rAcc, int N,
                     int iterations) {
  float *buf = new float[N * N];
  for (int i = 0; i < iterations; ++i) {
    addInFirst(sum, rAcc, N * N);
    multMatrices(N, r, rAcc, buf);
    std::copy(buf, buf + N * N, rAcc);
  }
  delete[] buf;
}
void tester(int N, int size, float *a, float *b, float *r, float *rAcc,
            float *sum, float *reverse) {
  auto start = std::chrono::steady_clock::now();

  auto_vectorizing::calcB(a, b, N);
  auto_vectorizing::calcR(a, b, r, N);
  std::copy(r, r + size, rAcc);
  auto_vectorizing::calculateSumRow(sum, r, rAcc, N, 10);
  auto_vectorizing::multMatrices(N, sum, b, reverse);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << std::setprecision(9)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
} // namespace auto_vectorizing

namespace with_intrinsics {
void multMatrices(int N, const float *A, const float *B, float *R) {
  for (int i = 0; i < N; ++i) {
    float *r = R + i * N;
    for (int j = 0; j < N; j += 8) {
      _mm256_storeu_ps(r + j, _mm256_setzero_ps());
    }
    for (int j = 0; j < N; ++j) {
      __m256 a = _mm256_set1_ps(A[i * N + j]);
      const float *b = B + j * N;
      for (int k = 0; k < N; k += 8) {
        _mm256_storeu_ps(r + k, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + k),
                                                _mm256_loadu_ps(r + k)));
      }
    }
  }
}
float findMaxRowSum(int N, const float *a) {
  float max = 0.0f;
  const __m256 mask = _mm256_set1_ps(-0.0f);
  for (int i = 0; i < N; ++i) {
    const float *ptr = a + i * N;
    __m256 s = _mm256_setzero_ps();
    for (int j = 0; j < N; j += 8) {
      s = _mm256_add_ps(_mm256_andnot_ps(mask, _mm256_loadu_ps(ptr + j)), s);
    }
    __m128 low = _mm256_castps256_ps128(s);
    __m128 high = _mm256_extractf128_ps(s, 1);
    low = _mm_add_ps(low, high);
    __m128 sh = _mm_shuffle_ps(low, low, _MM_SHUFFLE(3, 2, 3, 2));
    low = _mm_add_ps(low, sh);
    sh = _mm_shuffle_ps(low, low, _MM_SHUFFLE(0, 1, 0, 1));

    max = std::max(max, _mm_cvtss_f32(_mm_add_ss(low, sh)));
  }
  return max;
}
float findMaxColumnSum(int N, const float *a) {
  float *buf = new float[N];
  for (int i = 0; i < N; i += 8) {
    _mm256_storeu_ps(buf, _mm256_setzero_ps());
  }
  const __m256 mask = _mm256_set1_ps(-0.0f);
  for (int i = 0; i < N; ++i) {
    const float *ptr = a + i * N;
    for (int j = 0; j < N; j += 8) {
      __m256 row =
          _mm256_add_ps(_mm256_loadu_ps(buf + j),
                        _mm256_andnot_ps(mask, _mm256_loadu_ps(ptr + j)));
      _mm256_storeu_ps(buf + j, row);
    }
  }
  __m256 max = _mm256_loadu_ps(buf);
  for (int i = 8; i < N; i += 8) {
    max = _mm256_max_ps(_mm256_loadu_ps(buf + i), max);
  }
  float res = 0.0;
  for (int i = 0; i < 8; ++i) {
    res = std::max(res, max[i]);
  }
  delete[] buf;
  return res;
}
void calcB(const float *a, float *res, int N) {
  __m256 K =
      _mm256_set1_ps(1.0f / (findMaxRowSum(N, a) * findMaxColumnSum(N, a)));
  int blocksize = 8;
  for (int i = 0; i < N; i += blocksize) {
    for (int j = 0; j < N; j += blocksize) {
      // transpose the block beginning at [i,j]
      for (int k = i; k < i + blocksize; ++k) {
        __m256 aa = _mm256_mul_ps(_mm256_loadu_ps(&a[k * N]), K);
        for (int l = j; l < j + blocksize; ++l) {
          res[k + l * N] = aa[l % blocksize];
        }
      }
    }
  }
}
void calcR(const float *a, const float *b, float *r, const float *I, int N) {
  multMatrices(N, b, a, r);
  for (int i = 0; i < N; ++i) {
    float *rr = r + i * N;
    const float *ii = I + i * N;
    for (int j = 0; j < N; j += 8) {
      _mm256_storeu_ps(rr + j, _mm256_sub_ps(_mm256_loadu_ps(ii + j),
                                             _mm256_loadu_ps(rr + j)));
    }
  }
}
void calculateSumRow(float *sum, const float *r, float *rAcc, int N,
                     int iterations) {
  int size = N * N;
  float *buf = new float[size];
  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < size; j += 8) {
      _mm256_storeu_ps(sum + j, _mm256_add_ps(_mm256_loadu_ps(sum + j),
                                              _mm256_loadu_ps(rAcc + j)));
    }
    multMatrices(N, r, rAcc, buf);
    for (int j = 0; j < size; j += 8) {
      _mm256_storeu_ps(rAcc + j, _mm256_loadu_ps(buf + j));
    }
  }
  delete[] buf;
}
void tester(int N, int size, float *a, float *b, float *r, float *rAcc,
            float *sum, float *reverse, float *I) {
  std::cout << "hel";
  auto start = std::chrono::steady_clock::now();

  calcB(a, b, N);
  calcR(a, b, r, I, N);
  for (int i = 0; i < size; i += 8) {
    _mm256_storeu_ps(rAcc + i, _mm256_loadu_ps(r + i));
  }
  calculateSumRow(sum, r, rAcc, N, 10);
  multMatrices(N, sum, b, reverse);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << std::setprecision(9)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
} // namespace with_intrinsics

namespace with_blas {
void tester(int N, int size, float *a, float *r, float *rAcc, float *sum,
            float *reverse) {
  makeI(N, r);
  float *buf = new float[size];
  // std::cout << "BLAS" << std::endl;
  auto start = std::chrono::steady_clock::now();

  float max_row = 0.0f;
  for (int i = 0; i < N; ++i) {
    max_row = std::max(max_row, cblas_sasum(N, a + i * N, 1));
  }
  float max_col = with_intrinsics::findMaxColumnSum(N, a);
  float K = 1 / (max_row * max_col);
  /**
   *  void cblas_sgemm(const enum CBLAS_ORDER Order,
   *  const enum CBLAS_TRANSPOSE TransA,
   *  const enum CBLAS_TRANSPOSE TransB,
   *  const blasint M, const blasint N,
   *  const blasint K, const float alpha,
   *  const float *A, const blasint lda,
   *  const float *B, const blasint ldb,
   *  const float beta, float *C, const blasint ldc
   */
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, -K, a, N, a, N,
              1.0, r, N);

  for (int i = 0; i < size; i += 8) {
    _mm256_storeu_ps(rAcc + i, _mm256_loadu_ps(r + i));
  }

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < size; j += 8) {
      _mm256_storeu_ps(sum + j, _mm256_add_ps(_mm256_loadu_ps(sum + j),
                                              _mm256_loadu_ps(rAcc + j)));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, r, N,
                rAcc, N, 0.0, buf, N);
    for (int j = 0; j < size; j += 8) {
      _mm256_storeu_ps(rAcc + j, _mm256_loadu_ps(buf + j));
    }
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, K, sum, N, a, N,
              0.0, reverse, N);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << std::setprecision(9)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  delete[] buf;
}
} // namespace with_blas
int main() {
  const int N = 2048;

  int size = N * N;
  float *a = new float[size];
  initArray(a, size, "in_2048_2048.txt");

  float *b = new float[size];
  float *r = new float[size];
  float *rAcc = new float[size];

  float *sum = getArrayWithZeros(size);
  makeI(N, sum);

  float *I = getArrayWithZeros(size);
  makeI(N, I);

  float *reverse = new float[size];

//  for (int i = 0; i < 1; ++i) {
//    auto_vectorizing::tester(N, size, a, b, r, rAcc, sum, reverse);
//    //        with_intrinsics::tester(N, size, a, b, r, rAcc, sum, reverse, I);
//    //    with_blas::tester(N, size, a, r, rAcc, sum, reverse);
//    makeI(N, sum);
//    //    printArr(N, N, reverse);
//  }

  auto start = std::chrono::steady_clock::now();

    auto_vectorizing::calcB(a, b, N);
    auto_vectorizing::calcR(a, b, r, N);
    //std::copy(r, r + size, rAcc);
    //auto_vectorizing::calculateSumRow(sum, r, rAcc, N, 10);
    //auto_vectorizing::multMatrices(N, sum, b, reverse);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << std::setprecision(9)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

  delete[] a;
  delete[] b;
  delete[] r;
  delete[] rAcc;
  delete[] sum;
  delete[] reverse;
  delete[] I;

  return 0;
}

#pragma clang diagnostic pop
#pragma clang diagnostic pop