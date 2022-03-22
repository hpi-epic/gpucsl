// copy this macro to every file you use this in
#ifdef GTEST
#define CUDA_DEV
#define __device__
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#elif PYTHON_TEST
#define CUDA_DEV __global__
#else
#define CUDA_DEV __device__
#endif

// A[d1][d2] * B[d2][d3] = R[d1][d3]
template <int d1, int d2, int d3> CUDA_DEV void compute_matrix_multiplication(double *A, double *B, double *R) {
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d3; ++j) {
            double result = 0.0;
            for (int k = 0; k < d2; ++k) {
                result += A[i * d2 + k] * B[k * d3 + j];
            }
            R[i * d3 + j] = result;
        }
    }
}

// A^T[d1][d2] * B[d2][d3] = R[d1][d3]   -> A[d2][d1]
template <int d1, int d2, int d3>
CUDA_DEV void compute_matrix_multiplication_with_first_transposed(double *A, double *B, double *R) {
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d3; ++j) {
            double result = 0.0;
            for (int k = 0; k < d2; ++k) {
                result += A[k * d1 + i] * B[k * d3 + j];
            }
            R[i * d3 + j] = result;
        }
    }
}

// A[d1][d2] * B^T[d2][d3] = R[d1][d3]    -> B[d3][d2]
template <int d1, int d2, int d3>
CUDA_DEV void compute_matrix_multiplication_with_second_transposed(double *A, double *B, double *R) {
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d3; ++j) {
            double result = 0.0;
            for (int k = 0; k < d2; ++k) {
                result += A[i * d2 + k] * B[j * d2 + k];
            }
            R[i * d3 + j] = result;
        }
    }
}
template <int n> __device__ void fill_m0(int v_i, int v_j, double *M_0, double *C) {
    M_0[0 * 2 + 0] = 1.0;
    M_0[0 * 2 + 1] = C[v_i * n + v_j];
    M_0[1 * 2 + 0] = C[v_j * n + v_i];
    M_0[1 * 2 + 1] = 1.0;
}

template <int level, int n> __device__ void fill_m1(int v_i, int v_j, double *M_1, double *C, int *sepset) {
    for (int sepset_index = 0; sepset_index < level; ++sepset_index) {
        M_1[0 * level + sepset_index] = C[v_i * n + sepset[sepset_index]];
        M_1[1 * level + sepset_index] = C[v_j * n + sepset[sepset_index]];
    }
}

template <int level, int n> __device__ void fill_m2(double *M_2, double *C, int *sepset) {
    for (int i = 0; i < level; ++i) {
        for (int k = 0; k < level; ++k) {
            M_2[i * level + k] = C[sepset[i] * n + sepset[k]];
        }
    }
}

template <int level> __device__ void compute_h(double *H, double *M_0, double *M_1, double *M_2_inv) {
    const int H_dim = 2;
    double temp[level * 2];

    compute_matrix_multiplication_with_second_transposed<level, level, H_dim>(M_2_inv, M_1, temp);
    compute_matrix_multiplication<H_dim, level, H_dim>(M_1, temp, H);
    for (int i = 0; i < H_dim; ++i) {
        for (int j = 0; j < H_dim; ++j) {
            H[i * H_dim + j] = M_0[i * H_dim + j] - H[i * H_dim + j];
        }
    }
}

// Implementation of the following function (compute_inverse_matrix) is based on:
//   https://github.com/rahulmalhotra/Matrix_Inverse_Algo/blob/master/Code.cpp
// Original repository: https://github.com/rahulmalhotra/Matrix_Inverse_Algo
//   Branch: master
//   Commit: 3b982790b45b2a56f94e8feda093fc998c7929f5
// Original author: https://github.com/rahulmalhotra
// Original license:
// *****************************************************************************
// The MIT License (MIT)
//
// Copyright (c) 2016 Rahul Malhotra
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// *****************************************************************************
//
// ATTENTION: the original matrix gets modified!
template <int n> CUDA_DEV void compute_inverse_matrix(double *original_matrix, double *inversed_matrix) {
    int k;
    int j;
    int i;
    double p;
    // Set identity matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            inversed_matrix[i * n + j] = (i == j);
        }
    }

    for (i = 0; i < n; i++) {
        p = original_matrix[i * n + i];
        for (j = 0; j < n; j++) {
            inversed_matrix[i * n + j] = inversed_matrix[i * n + j] / p;
            original_matrix[i * n + j] = original_matrix[i * n + j] / p;
        }
        for (j = 0; j < n; j++) {
            p = original_matrix[j * n + i];
            for (k = 0; k < n; k++) {
                if (j != i) {
                    inversed_matrix[j * n + k] -= inversed_matrix[i * n + k] * p;
                    original_matrix[j * n + k] -= original_matrix[i * n + k] * p;
                }
            }
        }
    }
}

// Implementation of the following function (cholesky_decomposition) is based on:
//   https://github.com/olekscode/Cholesky-AVX/blob/master/src/chol.cpp
// Original repository: https://github.com/olekscode/Cholesky-AVX
//   Branch: master
//   Commit: 86963c6f7b4f2039005650ff29d5b787bd3a2358
// Original author: https://github.com/olekscode
// Original license:
// *****************************************************************************
// MIT License
//
// Copyright (c) 2018 Oleksandr Zaytsev
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// *****************************************************************************
template <int n> CUDA_DEV void cholesky_decomposition(double A[], double L[]) {
    memset(L, 0, sizeof(*L) * n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < (i + 1); ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                L[i * n + j] = sqrt(A[i * n + i] - sum);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - sum));
            }
        }
    }
}

// based on "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU" Algorithm 7
template <int level> CUDA_DEV void compute_m_2_inv(double *M_2, double *M_2_inv) {
    double L[level * level];
    double R[level * level];
    double temp[level * level];
    double *temp2 = M_2_inv; // Just an alias for readability; we use M_2_inv as a temporar storage location as well

    compute_matrix_multiplication_with_first_transposed<level, level, level>(M_2, M_2, temp);

    cholesky_decomposition<level>(temp, L);
    compute_matrix_multiplication_with_first_transposed<level, level, level>(L, L, temp);
    compute_inverse_matrix<level>(temp, R);

    compute_matrix_multiplication<level, level, level>(L, R, temp);
    compute_matrix_multiplication<level, level, level>(temp, R, temp2);
    compute_matrix_multiplication_with_second_transposed<level, level, level>(temp2, L, temp);
    compute_matrix_multiplication_with_second_transposed<level, level, level>(temp, M_2, M_2_inv);
}

__device__ double compute_fishers_z_transform(double *H) {
    const int H_dim = 2;

    double rho = H[0 * H_dim + 1] / (sqrt(fabs(H[0 * H_dim + 0])) * sqrt(fabs(H[1 * H_dim + 1])));
    double Z = fabs(0.5 * (log(fabs((1 + rho))) - log(fabs(1 - rho))));

    return Z;
}

// Based on: "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU", 2020
//   authors: Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi and Saber Salehkaleybar
//   journal: IEEE Transactions on Parallel and Distributed Systems (TPDS)
//   https://ieeexplore.ieee.org/document/8823064
// Parameters:
//     level: level of the CI test
//     n: number of variables in the dataset
//     v_i: first variable to perform CI test
//     v_j: second variable to perform CI test
//     C: correlation matrix of size n * n
//     sepset: separation set array of size level
template <int level, int n>
CUDA_DEV double gaussian_ci_test(unsigned int v_i, unsigned int v_j, double *C, int *sepset) {

    double M_0[2 * 2];
    double M_1[2 * level];
    double M_2[level * level];

    fill_m0<n>(v_i, v_j, M_0, C);
    fill_m1<level, n>(v_i, v_j, M_1, C, sepset);
    fill_m2<level, n>(M_2, C, sepset);

    double M_2_inv[level * level];

    compute_m_2_inv<level>(M_2, M_2_inv);

    const int H_dim = 2;
    double H[H_dim * H_dim];

    // H = M_1 - M_1 * M_2_inter = M_1 - M_1 * (M_2_inv * M_1_T)
    compute_h<level>(H, M_0, M_1, M_2_inv);

    return compute_fishers_z_transform(static_cast<double *>(H));
}
