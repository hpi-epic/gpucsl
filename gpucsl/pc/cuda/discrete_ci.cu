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

#include "helpers/discrete_helpers.cu"
#include "helpers/graph_helpers.cu"
#include "helpers/set_helpers.cu"

// Based on: "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data.", 2021
//   authors: Hagedorn, Christopher, and Johannes Huegle
//   journal: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)

// Template parameters and unused function parameters are just to keep the interface uniform for all levels
// Parameters:
//     -- Template parameters --
//     level: level of the CI test
//     n: number of variables in the dataset
//     max_dim: the maximum dimension of the variables in the dataset
//     n_observations: the number of observations in the dataset
//     -- Function Parameters --
//     data: the ordinal encoded discrete data in colum-major order of size n * n_observations
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     data_dimensions: the dimensions of the variables of size n
//     alpha: significance level for the conditional iindependence test
//     pmax: memory pointer to the datastructure for the maximum p values, size n*n
template <int level, int n, int, int max_dim, int n_observations, int>
__global__ void discrete_ci_level_0(uint8_t *data, unsigned short *G, int * /* unused */, uint8_t *data_dimensions,
                                    double alpha, uint32_t * /* unused */, int * /* unused */, float *pmax) {
    int v_i = blockIdx.x;
    int v_j = blockIdx.y;

    unsigned int v_i_ordered = v_i;
    unsigned int v_j_ordered = v_j;
    get_ordered_indices(&v_i_ordered, &v_j_ordered);

    // Remove diagonal edges from graph
    if (v_i == v_j && v_i < n) {
        G[v_i * n + v_j] = 0;
        G[v_j * n + v_i] = 0;
    }

    if (v_j >= v_i) {
        return;
    }
    __shared__ uint32_t contingency_matrix[max_dim * max_dim];
    for (int contingency_index = threadIdx.x; contingency_index < max_dim * max_dim; contingency_index += blockDim.x) {
        contingency_matrix[contingency_index] = 0;
    }
    __syncthreads();

    calculate_contingency_matrix_level_0<n_observations>(&data[v_i * n_observations], &data[v_j * n_observations],
                                                         contingency_matrix, data_dimensions[v_j]);
    __syncthreads();

    if (threadIdx.x == 0) {
        int dim_v_i = data_dimensions[v_i];
        int dim_v_j = data_dimensions[v_j];

        int marginals_v_i[max_dim];
        int marginals_v_j[max_dim];
        for (int index = 0; index < max_dim; index++) {
            marginals_v_i[index] = 0;
            marginals_v_j[index] = 0;
        }

        calculate_marginals_level_0(contingency_matrix, dim_v_i, dim_v_j, marginals_v_i, marginals_v_j);
        // Discrete CI test
        // 1. Calculate chi_squared
        double chi_squared = 0;
        calculate_chi_squared_level_0<n_observations>(&chi_squared, dim_v_i, dim_v_j, contingency_matrix, marginals_v_i,
                                                      marginals_v_j);

        // 2. Update edges based on outcome
        double pvalue = pchisq(chi_squared, (dim_v_i - 1) * (dim_v_j - 1));

        write_atomic_maximum(&pmax[v_j_ordered * n + v_i_ordered], (float)pvalue);

        if (pvalue >= alpha) {
            G[v_i * n + v_j] = 0;
            G[v_j * n + v_i] = 0;
        } else {
            G[v_i * n + v_j] = 1;
            G[v_j * n + v_i] = 1;
        }
    }
}

// Based on: "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data.", 2021
//   authors: Hagedorn, Christopher, and Johannes Huegle
//   journal: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)
// Helper function to execute ci test for specific seperation set
//     -- Template parameters --
//     level: level of the CI test
//     n_observations: the number of observations in the dataset
//     -- Function Parameters --
//     v_i, v_j: indexes of variables
//     seperation_variables: array of length level of indexes of seperation variabes
//     data_dimensions: the dimensions of the variables of size n
//     thread_working_memory: memory pointer to datastructure for contingency table and marginal tables
//                            offset in a way that this data is unique to the thread
//     pvalue[OUTPUT]: result of the conditional independence test
template <int level, int n_observations, int parallel_ci_tests>
__global__ void execute_ci_test_for_sepset(int v_i, int v_j, int *seperation_variables, uint8_t *data_dimensions,
                                           uint32_t *thread_working_memory, uint8_t *data, float *pvalue) {

    int dim_v_i = data_dimensions[v_i];
    int dim_v_j = data_dimensions[v_j];

    int dim_s = 1;
    for (int x = 0; x < level; x++) {
        dim_s *= data_dimensions[seperation_variables[x]];
    }

    uint32_t *N_vi_vj_s;
    uint32_t *N_plus_vj_s;
    uint32_t *N_vi_plus_s;
    uint32_t *N_plus_plus_s;

    // This happens multiple times, as there are blockDim.y * blockDim.z threads with threadIdx.x == 0
    N_vi_vj_s = thread_working_memory;
    N_plus_vj_s = &N_vi_vj_s[dim_s * dim_v_i * dim_v_j];
    N_vi_plus_s = &N_plus_vj_s[dim_s * dim_v_j];
    N_plus_plus_s = &N_vi_plus_s[dim_s * dim_v_i];

    if (threadIdx.x == 0) {
        uint32_t malloc_size = dim_s * dim_v_i * dim_v_j // N_vi_vj_s |N_+_+_S| * |v_i| * |v_j|
                               + dim_s * dim_v_j         // N_+_vj_s |N_+_+_S| * |v_j|
                               + dim_s * dim_v_i         // N_vi_+_s  |N_+_+_s| * |v_i|
                               + dim_s;                  // N_+_+_s  Multiply: |s| for all s in S
        memset(N_vi_vj_s, 0, malloc_size * sizeof(uint32_t));
    }
    __syncthreads();
    calculate_contingency_matrix_level_n<n_observations, level>(data, N_vi_vj_s, v_i, v_j, seperation_variables,
                                                                data_dimensions);
    __syncthreads();
    calculate_marginals_level_n(N_vi_vj_s, N_vi_plus_s, N_plus_vj_s, N_plus_plus_s, dim_v_i, dim_v_j, dim_s);
    __syncthreads();

    // Calculate local statistics
    __shared__ double chi_squared[parallel_ci_tests];
    double *local_chi_squared = &chi_squared[blockDim.z * threadIdx.y + threadIdx.z];
    *local_chi_squared = 0;
    __syncthreads();
    calculate_chi_squared_level_n(local_chi_squared, N_vi_vj_s, N_vi_plus_s, N_plus_vj_s, N_plus_plus_s, dim_s, dim_v_i,
                                  dim_v_j);
    __syncthreads();

    if (threadIdx.x == 0) {
        *pvalue = pchisq(*local_chi_squared, (dim_v_i - 1) * (dim_v_j - 1) * dim_s);
    }
    __syncthreads();
}

// Based on: "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data.", 2021
//   authors: Hagedorn, Christopher, and Johannes Huegle
//   journal: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)

// Template parameters and unused function parameters are just to keep the interface uniform for all levels
// Parameters:
//     -- Template parameters --
//     level: level of the CI test
//     n: number of variables in the dataset
//     parallel_ci_tests: number of conditional independence tests that happen concurrently in one block
//     max_dim: the maximum dimension of the variables in the dataset
//     n_observations: the number of observations in the dataset
//     max_level: the maximum execution level
//     -- Function Parameters --
//     data: the ordinal encoded discrete data in colum-major order of size n * n_observations
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     data_dimensions: the dimensions of the variables of size n
//     alpha: significance level for the conditional iindependence test
//     working_memory: memory pointer to datastructure for contingency table and marginal tables
//     pmax: memory pointer to the datastructure for the maximum p values
template <int level, int n, int parallel_ci_tests, int max_dim, int n_observations, int max_level>
__global__ void discrete_ci_level_n(uint8_t *data, unsigned short *G, int *G_compacted, uint8_t *data_dimensions,
                                    double alpha, uint32_t *working_memory, int *separation_sets, float *pmax) {

    // process memory_reduction_factor many logical blocks sequentially in one actual block
    int memory_reduction_factor = (n + gridDim.x - 1) / gridDim.x;

    for (int v_i_offset = 0; v_i_offset < memory_reduction_factor; v_i_offset++) {

        int v_i = blockIdx.x * memory_reduction_factor + v_i_offset;

        if (v_i >= n) {
            continue;
        }

        // Calculate the space for the contingency table and 3 different marginal tables
        uint32_t max_dim_s = pow((double)max_dim, (double)level);
        uint32_t reserved_size_per_ci_test = max_dim_s * max_dim * max_dim + 2 * max_dim_s * max_dim + max_dim_s;

        int neighbours_count;
        int *neighbours;
        get_neighbours<n>(v_i, G_compacted, &neighbours_count, &neighbours);

        int idx_v_j = blockIdx.y * blockDim.z + threadIdx.z;
        if (idx_v_j >= neighbours_count) {
            // Early return, as this block can not have a valid v_j
            continue;
        }
        int v_j = neighbours[idx_v_j];

        unsigned int v_i_ordered = v_i;
        unsigned int v_j_ordered = v_j;
        get_ordered_indices(&v_i_ordered, &v_j_ordered);

        uint32_t sepset_count = binomial_coefficient(neighbours_count - 1, level);

        for (int sepset_index = threadIdx.y; sepset_index < sepset_count; sepset_index += blockDim.y) {

            // Generate unique index for each unique blockIdx.x,blockIdx.y, threadIdx.y, and threadIdx.z
            // Memory is shared between different threadIdx.x, as these work together on one CI-Test
            int thread_memory_index = ((gridDim.y * blockDim.y * blockDim.z) * blockIdx.x +
                                       (blockDim.y * blockDim.z) * blockIdx.y + blockDim.y * threadIdx.z + threadIdx.y);
            uint32_t *thread_working_memory = &working_memory[reserved_size_per_ci_test * thread_memory_index];
            float pvalue;
            int seperation_variables[level];
            get_sepset<level>(neighbours, neighbours_count - 1, sepset_index, seperation_variables, v_j);
            execute_ci_test_for_sepset<level, n_observations, parallel_ci_tests>(
                v_i, v_j, seperation_variables, data_dimensions, thread_working_memory, data, &pvalue);

            if (threadIdx.x == 0) {
                write_atomic_maximum(&pmax[v_j_ordered * n + v_i_ordered], pvalue);
                if (pvalue >= alpha) {
                    remove_edge_synced<level, n, max_level>(v_i, v_j, G, separation_sets, seperation_variables);
                }
            }
            __syncthreads();
            if (G[v_i * n + v_j] == 0) {
                continue;
            }
        }
    }
}
