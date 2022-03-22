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

#include <stdint.h>

// Calculates the contingency matrix for two variables
//    column_a, column_b: data vectors for variables
//    dimension_a, dimension_b: respective dimensions
//    n_observations: number of observations
//    result: contingency matrix of size dimension_a * dimension_b. result[value_a * dimension_b + value_b]
//         gives the number of occurrences for value_a in v_a and value_b in v_b
template <int n_observations>
CUDA_DEV void calculate_contingency_matrix_level_0(uint8_t *column_a, uint8_t *column_b, uint32_t *result,
                                                   int dimension_b) {
    int thread_count = blockDim.x;
    for (int k = threadIdx.x; k < n_observations; k += thread_count) {
        atomicAdd(&result[column_a[k] * dimension_b + column_b[k]], 1);
    }
}

// Calculates an index that can be used to store data to a specific location for one given seperation set data point
// (This index is unique for a specific variable allocation from the seperation variables)
//    data: the ordinal encoded discrete data in colum-major order of size n * n_observations
//    seperation_variables: indexes for the variables in the seperation sets
//    data_dimensions: the dimensions of the variables of size n
//    [OUTPUT]: index
template <int level, int n_observations>
__device__ int calculate_sepset_index(int data_index, uint8_t *data, int *separation_variables,
                                      uint8_t *data_dimensions) {
    int sum_sep = 0;
    int cat_sep = 1;
    for (int s = 0; s < level; s++) {
        int vs = separation_variables[s];
        sum_sep += data[vs * n_observations + data_index] * cat_sep;
        cat_sep *= data_dimensions[vs];
    }
    return sum_sep;
}

// Helper function to calculate the index into N_vi_vj_s
//   sepset_index: index of seperation set, can be calculated with calculate_sepset_index
//   dim_v_i, dim_v_j: dimensions of variables v_i, v_j
//   value_v_i, value_v_j: values for variables v_i, v_j
__device__ int get_n_vi_vj_s_index(int sepset_index, int dim_v_i, int dim_v_j, int value_v_i, int value_v_j) {
    return sepset_index * dim_v_i * dim_v_j + value_v_i * dim_v_j + value_v_j;
}

// Calculates the contingency matrix for two variables and given seperation set
//    data: the ordinal encoded discrete data in colum-major order of size n * n_observations
//    N_vi_vj_s[OUTPUT]: contingency matrix of size dim_v_i * dim_v_j * dim_s that stores the
//                       number of occurences for each combination and sepset
//    v_i, v_j: index of variables for the data
//    seperation_variables: indexes for the variables in the seperation sets
//    data_dimensions: the dimensions of the variables of size n
template <int n_observations, int level>
CUDA_DEV void calculate_contingency_matrix_level_n(uint8_t *data, uint32_t *N_vi_vj_s, int v_i, int v_j,
                                                   int *seperation_variables, uint8_t *data_dimensions) {
    int dim_v_i = data_dimensions[v_i];
    int dim_v_j = data_dimensions[v_j];

    for (int data_index = threadIdx.x; data_index < n_observations; data_index += blockDim.x) {
        int value_v_i = data[v_i * n_observations + data_index];
        int value_v_j = data[v_j * n_observations + data_index];

        int sepset_index =
            calculate_sepset_index<level, n_observations>(data_index, data, seperation_variables, data_dimensions);

        int index = get_n_vi_vj_s_index(sepset_index, dim_v_i, dim_v_j, value_v_i, value_v_j);

        atomicAdd(&N_vi_vj_s[index], 1);
    }
}

// Calculates the marginal vectors for two variables
// For more details about the notation of marginal and contingency table consider the paper
//          "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data.", 2021
//          authors: Hagedorn, Christopher, and Johannes Huegle
//          journal: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)
//
//    contingency_matrix: matrix that stores the number of occurrences for each combination and sepset
//    dim_v_i, dim_v_j: dimensions of v_i and v_j
//    marginals_v_i, marginals_v_j: marginal vectors for v_i and v_j, storing number of specific occourences of given
//                                  value
CUDA_DEV void calculate_marginals_level_0(uint32_t *contingency_matrix, int dim_v_i, int dim_v_j, int *marginals_v_i,
                                          int *marginals_v_j) {
    for (int v_i_value = 0; v_i_value < dim_v_i; v_i_value++) {
        for (int v_j_value = 0; v_j_value < dim_v_j; v_j_value++) {
            int entry = contingency_matrix[v_i_value * dim_v_j + v_j_value];
            marginals_v_i[v_i_value] += entry;
            marginals_v_j[v_j_value] += entry;
        }
    }
}

// Calculates the marginal matrices for two variables and given seperation set
// For more details about the notation of marginal and contingency table consider the paper
//          "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data.", 2021
//          authors: Hagedorn, Christopher, and Johannes Huegle
//          journal: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)
//
//    N_vi_vj_s: contingency matrix that stores the number of occurrences for each combination and sepset
//    N_vi_plus_s[OUTPUT]: marginal vector for given v_i and sepset
//    N_plus_vj_s[OUTPUT]: marginal vector for given v_j and sepset
//    N_plus_plus_s[OUTPUT]: marginal vector for given sepset
//    dim_v_i, dim_v_j: dimensions of v_i and v_j
//    dim_s: dimension of sepset, which is the product of single dimensions of the sepset variables
CUDA_DEV void calculate_marginals_level_n(uint32_t *N_vi_vj_s, uint32_t *N_vi_plus_s, uint32_t *N_plus_vj_s,
                                          uint32_t *N_plus_plus_s, uint8_t dim_v_i, uint8_t dim_v_j, int dim_s) {
    for (int g = threadIdx.x; g < dim_s; g += blockDim.x) {
        for (int v_i_value = 0; v_i_value < dim_v_i; v_i_value++) {
            for (int v_j_value = 0; v_j_value < dim_v_j; v_j_value++) {
                uint32_t observed = N_vi_vj_s[get_n_vi_vj_s_index(g, dim_v_i, dim_v_j, v_i_value, v_j_value)];
                // printf("obs: %u\n", observed);
                atomicAdd(&N_vi_plus_s[g * dim_v_i + v_i_value], observed);
                atomicAdd(&N_plus_vj_s[g * dim_v_j + v_j_value], observed);
                atomicAdd(&N_plus_plus_s[g], observed);
            }
        }
    }
}

// Calculate the chi_squared value given the marginal matrices without sepset
// chi_squared[OUTPUT]
// dim_v_i, dim_v_j: dimensions of variables v_i and v_j
// contingency_matrix: contingency matrix between v_i and v_j, as generated by calculate_contingency_matrix_level_0
// marginals_v_i, marginals_v_j: marginal vectors for v_i and v_j
template <int n_observations>
CUDA_DEV void calculate_chi_squared_level_0(double *chi_squared, int dim_v_i, int dim_v_j, uint32_t *contingency_matrix,
                                            int *marginals_v_i, int *marginals_v_j) {
    double expected = 0;
    double observed = 0;
    for (int v_i_value = 0; v_i_value < dim_v_i; v_i_value++) {
        for (int v_j_value = 0; v_j_value < dim_v_j; v_j_value++) {
            expected = marginals_v_i[v_i_value] * (double)marginals_v_j[v_j_value] / n_observations;
            // printf("expected: %f\n", expected);
            if (expected != 0) {
                observed = (double)contingency_matrix[v_i_value * dim_v_j + v_j_value];
                *chi_squared += (observed - expected) * (observed - expected) / expected;
            }
        }
    }
}

// Calculate the chi_squared value given the marginal matrices with sepset
//    chi_squared[OUTPUT]
//    N_vi_vj_s, N_vi_plus_s, N_plus_vj_s, N_plus_plus: contingency and marginal matrices as generated by
//                                                   calculate_contingency_matrix_level_n  and
//                                                   calculate_marginals_level_n
//    dim_v_i, dim_v_j: dimensions of v_i and v_j
//    dim_s: dimension of sepset, which is the product of single dimensions of the sepset variables
CUDA_DEV void calculate_chi_squared_level_n(double *output, uint32_t *N_vi_vj_s, uint32_t *N_vi_plus_s,
                                            uint32_t *N_plus_vj_s, uint32_t *N_plus_plus_s, int dim_s, int dim_v_i,
                                            int dim_v_j) {
    *output = 0.0;
    __syncthreads();
    for (int g = threadIdx.x; g < dim_s; g += blockDim.x) {
        for (int v_i = 0; v_i < dim_v_i; v_i++) {
            for (int v_j = 0; v_j < dim_v_j; v_j++) {
                uint32_t observed = N_vi_vj_s[g * dim_v_i * dim_v_j + v_i * dim_v_j + v_j];

                double expected =
                    N_vi_plus_s[g * dim_v_i + v_i] * (double)N_plus_vj_s[g * dim_v_j + v_j] / N_plus_plus_s[g];
                if (N_plus_plus_s[g] == 0 || expected == 0.0) {
                    continue;
                }

                double sum_term = (observed - expected) * (observed - expected) / expected;
                atomicAdd(output, sum_term);
            }
        }
    }
    __syncthreads();
}
