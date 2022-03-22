// copy this macro to every file you use this in
#ifdef GTEST
#define CUDA_DEV
#define __device__
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#elif PYTHON_TEST
#define CUDA_DEV __global__
#else
#define CUDA_DEV __device__
#endif

#include <cassert>

#include "math_helpers.cu"

// Based on: "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU", 2020
//   authors: Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi and Saber Salehkaleybar
//   journal: IEEE Transactions on Parallel and Distributed Systems (TPDS)
//   https://ieeexplore.ieee.org/document/8823064
template <int level>
CUDA_DEV void calculate_sepset_indices(int neighbours_count, int sepset_index, int *sepset_indices) {
    int sum = 0;

    // First iteration
    sepset_indices[0] = 0;

    while (sum <= sepset_index) {
        sepset_indices[0]++;
        sum += binomial_coefficient(neighbours_count - sepset_indices[0], level - (0 + 1));
    }

    sum -= binomial_coefficient(neighbours_count - sepset_indices[0], level - (0 + 1));

    // Following iterations
    for (int c = 1; c < level; c++) {
        sepset_indices[c] = sepset_indices[c - 1];

        while (sum <= sepset_index) {
            sepset_indices[c]++;
            sum += binomial_coefficient(neighbours_count - sepset_indices[c], level - (c + 1));
        }
        sum -= binomial_coefficient(neighbours_count - sepset_indices[c], level - (c + 1));
    }
}

template <int level> CUDA_DEV void resolve_sepset_indices(int *neighbours, int *sepset, int v_j) {
    for (int i = 0; i < level; ++i) {
        int index = sepset[i] - 1;
        index += neighbours[index] >= v_j;
        sepset[i] = neighbours[index];
    }
}

// in: sepset[LEVEL]
// out: sepset[LEVEL] array with indices of seperation_variables
template <int level>
CUDA_DEV void get_sepset(int *neighbours, int neighbours_count, int sepset_index, int *sepset, int v_j) {
    calculate_sepset_indices<level>(neighbours_count, sepset_index, sepset);
#ifdef __CUDACC_DEBUG__
    for (int s_index = 0; s_index < level; ++s_index) {
        assert(sepset[s_index] <= neighbours_count);
        assert(sepset[s_index] > 0);
    }
#endif
    resolve_sepset_indices<level>(neighbours, sepset, v_j);
}
