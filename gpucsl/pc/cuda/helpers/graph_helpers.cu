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

// Compacts the adjecency matrix; the first entry in the ith row of the output matrix is the count of edges
// of the variable i and the following entries represent the indexes of the neighbour nodes of i.
// Example:
// Adjacency matrix(G)        Compacted matrix(G_compacted)
// [0, 1, 1, 0, 0, 1]   =>   [3, 1, 2, 5, 0, 0]
// [1, 0, 1, 0, 1, 0]   =>   [3, 0, 2, 4, 0, 0]
// [0, 0, 0, 0, 0, 0]   =>   [0, 0, 0, 0, 0, 0]
// [1, 1, 1, 0, 1, 1]   =>   [5, 0, 1, 2, 4, 5]
// [1, 0, 1, 0, 0, 1]   =>   [3, 0, 2, 5, 0, 0]
// [0, 1, 1, 1, 0, 0]   =>   [3, 1, 2, 3, 0, 0]
// Parameters:
//     n: number of variables in the dataset
//     columns_per_device: number of columns needed to be compacted per device
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     G_compacted: compacted adjacency matrix of G; see compact function for layout
//     device_index: index of the device the kernel is executed on
template <int n, unsigned int columns_per_device>
__global__ void compact(unsigned short *G, unsigned int *G_compacted, unsigned int device_index) {
    unsigned int offset = device_index * columns_per_device;
    unsigned int local_column = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_column >= columns_per_device) {
        return;
    }

    unsigned int column = offset + local_column;
    unsigned int count = 0;

    if (column < n) {
        for (int i = 0; i < n; i++) {
            char edge_exists = G[column * n + i] * (i != column);
            count += edge_exists;
            G_compacted[column * n + (count * edge_exists)] = i;
        }

        G_compacted[column * n + 0] = count;
    }
}

// Get neighbours from compacted graph
// n_variables: number of variables ingraph
// -----------
// v: index of variable to seardch neighbours for
// G_compacted: compacted graph from `compact` function
// neighbours_count[OUTPUT]: number of neighbours
// neighbours[OUTPUT]: indexes of neighbour nodes
template <int n_variables>
__device__ void get_neighbours(unsigned int v, int *G_compacted, int *neighbours_count, int **neighbours) {
    *neighbours_count = G_compacted[v * n_variables];
    *neighbours = &G_compacted[v * n_variables + 1];
}

// Deletes an edge from the graph and stores the corresponding seperation set and p value
// Prevents race conditions by using the smaller edge in the graph data structure as a mutex
//     level: level of the CI test
//     n: number of variables in the dataset
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     separation_sets: memory pointer for seperation_set data structure
//     seperation_set: separation set to store. Length is implicitly given by level, as sepearation_set is always level
//                     long
//     pmax: memory pointer to pmax data structure
//     pvalue: pvalue to store into pmax structure
template <int level, int n, int max_level>
CUDA_DEV void remove_edge_synced(int v_i, int v_j, unsigned short *G, int *separation_sets, int *separation_set) {
    // swap v_i and v_j so we do not run into race conditions that could appear if two threads at the same
    // time enter the next condition
    if (v_i < v_j) {
        unsigned int temp = v_i;
        v_i = v_j;
        v_j = temp;
    }

    // IMPORTANT: if not already G[v_i * n + v_j] = 0 it will be set in the condition of the if clause
    // also works as a mutex so only one thread of a block can enter the condition at the same time
    if (atomicCAS(&G[v_i * n + v_j], (unsigned short)1, (unsigned short)0) == 1) {
        G[v_j * n + v_i] = 0;

        for (int i = 0; i < level; i++) {
            separation_sets[(v_i * n + v_j) * max_level + i] = separation_set[i];
            separation_sets[(v_j * n + v_i) * max_level + i] = separation_set[i];
        }
    }
}

CUDA_DEV void get_ordered_indices(unsigned int *v_i, unsigned int *v_j) {
    if (*v_i < *v_j) {
        unsigned int temp = *v_i;
        *v_i = *v_j;
        *v_j = temp;
    }
}

__device__ float write_atomic_maximum(float *pmax, float value) {
    int *int_pmax = (int *)pmax; // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    int old_value_int = *int_pmax;
    int new_value_int;
    do {
        new_value_int = old_value_int;
        old_value_int =
            atomicCAS(int_pmax, new_value_int, __float_as_int(::fmaxf(value, __int_as_float(new_value_int))));
    } while (new_value_int != old_value_int);
    return __int_as_float(old_value_int);
}

__device__ float write_atomic_minimum(float *pmax, float value) {
    int *int_pmax = (int *)pmax; // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    int old_value_int = *int_pmax;
    int new_value_int;
    do {
        new_value_int = old_value_int;
        old_value_int =
            atomicCAS(int_pmax, new_value_int, __float_as_int(::fminf(value, __int_as_float(new_value_int))));
    } while (new_value_int != old_value_int);
    return __int_as_float(old_value_int);
}
