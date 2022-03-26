#include "helpers/gaussian_helpers.cu"
#include "helpers/graph_helpers.cu"
#include "helpers/set_helpers.cu"

// Based on: "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU", 2020
//   authors: Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi and Saber Salehkaleybar
//   journal: IEEE Transactions on Parallel and Distributed Systems (TPDS)
//   https://ieeexplore.ieee.org/document/8823064
// Template parameters and unused function parameters are just to keep the interface uniform for all levels
// Parameters:
//     level: level of the CI test
//     n: number of variables in the dataset
//     C: correlation matrix of size n * n
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     threshold: threshold value used to perform the CI test
//     zmin: n*n matrix to store minimum fishers z value per edge (has to be transformed to pmax by the caller)
//     device_index: index of the device the kernel is executed on
template <int level, int n, int max_level>
__global__ void gaussian_ci_level_0(double *C, unsigned short *G, int * /*unused*/, double threshold, int * /*unused*/,
                                    float *zmin, int /*device_index*/) {
    // We only calculate one half of the adjacency matrix (as it is symmetric and we can just set the other half).
    // Thus there are sum of 1 to n ((n * (n + 1)) / 2) many values to calculate and we map them directly to threads.
    // Therefore every thread has an index i with 0 <= i < (n * (n + 1)) / 2. This gets mapped to the column (v_i)
    // index by calculating the inverse of the sum of integers (for example threads with the ids 10 - 14 calculate on
    // column 4). Then we get the row (v_j) by calculating the difference of its id to the next smaller sum of integers
    // (for id 13 -> column = 4 -> next smaller sum of intergers = 10 -> row = 3).
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v_i = sqrt(2 * id + 0.25) - 0.5;
    unsigned int v_j = id - ((v_i * (v_i + 1) / 2));

    unsigned int v_i_ordered = v_i;
    unsigned int v_j_ordered = v_j;
    get_ordered_indices(&v_i_ordered, &v_j_ordered);

    if (v_i < n && v_j < n) {
        double Z = C[v_j * n + v_i];
        Z = abs(0.5 * log(abs((1 + Z) / (1 - Z))));
        write_atomic_minimum(&zmin[v_j_ordered * n + v_i_ordered], (float)Z);
        if (Z < threshold) {
            G[v_j * n + v_i] = 0;
            G[v_i * n + v_j] = 0;
        } else {
            G[v_j * n + v_i] = 1;
            G[v_i * n + v_j] = 1;
        }
    }
    if (v_j == v_i && v_i < n) {
        G[v_j * n + v_i] = 0;
        G[v_i * n + v_j] = 0;
    }
}

// Based on: "cuPC: CUDA-based Parallel PC Algorithm for Causal Structure Learning on GPU", 2020
//   authors: Behrooz Zarebavani, Foad Jafarinejad, Matin Hashemi and Saber Salehkaleybar
//   journal: IEEE Transactions on Parallel and Distributed Systems (TPDS)
//   https://ieeexplore.ieee.org/document/8823064
// Parameters:
//     level: level of the CI test
//     n: number of variables in the dataset
//     C: correlation matrix of size n * n
//     G: adjacency matrix representation of the dependency graph (1: edge, 0: no edge) of size n * n
//     G_compacted: compacted adjacency matrix of G; see compact function for layout
//     threshold: threshold value used to perform the CI test
//     separation_sets: variable used to return found seperation sets
//     device_index: index of the device the kernel is executed on
template <int level, int n, int max_level>
__global__ void gaussian_ci_level_n(double *C, unsigned short *G, int *G_compacted, double threshold,
                                    int *separation_sets, float *zmin, int device_index) {

    unsigned int v_i = gridDim.x * device_index + blockIdx.x;
    unsigned int v_j = blockIdx.y;

    if (v_i >= n || v_j >= n) {
        return;
    }

    if (v_i == v_j) {
        return;
    }

    // swap v_i and v_j so we do not run into race conditions that could appear if two threads at the same
    // time enter the next condition
    unsigned int v_i_ordered = v_i;
    unsigned int v_j_ordered = v_j;
    get_ordered_indices(&v_i_ordered, &v_j_ordered);

    int neighbours_count;
    int *neighbours;
    get_neighbours<n>(v_i, G_compacted, &neighbours_count, &neighbours);

    int sepset_neighbours_count = max(neighbours_count - 1, 0); // v_j cannot be in a separation set

    if (sepset_neighbours_count < level) {
        return;
    }

    int sepset_count = binomial_coefficient(sepset_neighbours_count, level);

    for (unsigned int sepset_index = threadIdx.x; sepset_index < sepset_count; sepset_index += blockDim.x) {
        int sepset[level];
        calculate_sepset_indices<level>(sepset_neighbours_count, sepset_index, sepset);
        for (int s_index = 0; s_index < level; ++s_index) {
            assert(sepset[s_index] <= neighbours_count);
            assert(sepset[s_index] > 0);
        }
        resolve_sepset_indices<level>(neighbours, sepset, v_j);

        double Z = gaussian_ci_test<level, n>(v_i, v_j, C, sepset);
        write_atomic_minimum(&zmin[v_j_ordered * n + v_i_ordered], (float)Z); // only fill upper half

        if (Z < threshold) {
            remove_edge_synced<level, n, max_level>(v_i, v_j, G, separation_sets, sepset);
        }

        // Early termination
        if (G[v_i * n + v_j] == 0) {
            break;
        }
    }
}
