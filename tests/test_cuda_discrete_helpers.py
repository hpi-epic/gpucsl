import cupy as cp
import numpy as np
import pandas as pd
import itertools
import math
import networkx as nx
from gpucsl.pc.kernel_management import get_module

function_names = [
    "calculate_contingency_matrix_level_0<12>",
    "calculate_contingency_matrix_level_n<100, 2>",
    "calculate_marginals_level_n",
    "calculate_chi_squared_level_n",
]
module = get_module(
    "helpers/discrete_helpers.cu", function_names, ("-D", "PYTHON_TEST")
)


def test_calculate_contingency_matrix_level_n():
    kernel = module.get_function("calculate_contingency_matrix_level_0<12>")

    # Random data points for two variables
    data_a = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6], int)
    data_b = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0], int)

    expected_result = pd.crosstab(data_a, data_b).to_numpy()

    d_data_a = cp.asarray(data_a, cp.uint8)
    d_data_b = cp.asarray(data_b, cp.uint8)

    d_result = cp.zeros((7, 2), cp.uint32)

    n_parallel_threads = 10

    kernel(
        (1,),
        (n_parallel_threads,),
        (d_data_a, d_data_b, d_result, cp.int32(2)),
    )
    assert cp.isclose(expected_result, d_result).all()


def test_calculate_contingency_matrix_level_n_random_values():
    n_observations = 100
    level = 2
    kernel = module.get_function(
        f"calculate_contingency_matrix_level_n<{n_observations}, {level}>"
    )
    np.random.seed(1)
    # Random data points for 5 seperation variables and 2 variables (vi, vj)
    data = np.random.randint(0, 5, size=(n_observations, 4))
    print(data)

    d_vi = cp.int32(2)
    d_vj = cp.int32(3)
    d_seperation_variables = cp.asarray([0, 1], dtype=cp.int32)

    expected_counts = np.zeros((5, 5, 5, 5))

    for row in data:
        expected_counts[row[1], row[0], row[2], row[3]] += 1

    d_data = cp.asarray(data, order="F", dtype=cp.uint8)
    d_dims = cp.full(4, 5, dtype=cp.uint8)

    d_result = cp.zeros(pow(5, 4), cp.uint32)

    kernel(
        (1,),
        (1,),
        (
            d_data,
            d_result,
            d_vi,
            d_vj,
            d_seperation_variables,
            d_dims,
            2,
        ),
    )

    assert cp.isclose(expected_counts.flatten(), d_result).all()


def test_calculate_marginals_level_n():
    kernel = module.get_function("calculate_marginals_level_n")
    dim_vi = 5
    dim_vj = 5
    dim_s = 5

    np.random.seed(1)
    N_vi_vj_s = np.random.randint(1, 100, size=(dim_vi, dim_vj, dim_s))
    N_plus_plus_s = np.zeros(5)
    N_vi_plus_s = np.zeros((5, 5))
    N_plus_vj_s = np.zeros((5, 5))

    # Generate marginals
    for s, vi, vj in itertools.product(range(5), repeat=3):
        value = N_vi_vj_s[s, vi, vj]
        N_plus_plus_s[s] += value
        N_vi_plus_s[s, vi] += value
        N_plus_vj_s[s, vj] += value

    d_N_vi_plus_s = cp.zeros_like(N_vi_plus_s, dtype=cp.uint32)
    d_N_plus_vj_s = cp.zeros_like(N_plus_vj_s, dtype=cp.uint32)
    d_N_plus_plus_s = cp.zeros_like(N_plus_plus_s, dtype=cp.uint32)

    kernel(
        (1,),
        (32,),
        (
            cp.asarray(N_vi_vj_s, dtype=cp.uint32),
            cp.asarray(d_N_vi_plus_s, dtype=cp.uint32),
            cp.asarray(d_N_plus_vj_s, dtype=cp.uint32),
            cp.asarray(d_N_plus_plus_s, dtype=cp.uint32),
            cp.uint8(5),
            cp.uint8(5),
            cp.int32(5),
        ),
    )

    assert cp.isclose(d_N_vi_plus_s, N_vi_plus_s).all()
    assert cp.isclose(d_N_plus_vj_s, N_plus_vj_s).all()
    assert cp.isclose(d_N_plus_plus_s, N_plus_plus_s).all()


def test_calculate_calculated_chi_squared():
    kernel = module.get_function("calculate_chi_squared_level_n")

    dim_vi = 5
    dim_vj = 5
    dim_s = 5

    np.random.seed(1)
    N_vi_vj_s = np.random.randint(1, 100, size=(dim_vi, dim_vj, dim_s))
    N_plus_plus_s = np.zeros(5)
    N_vi_plus_s = np.zeros((5, 5))
    N_plus_vj_s = np.zeros((5, 5))

    # Generate marginals
    for s, vi, vj in itertools.product(range(5), repeat=3):
        value = N_vi_vj_s[s, vi, vj]
        N_plus_plus_s[s] += value
        N_vi_plus_s[s, vi] += value
        N_plus_vj_s[s, vj] += value

    # Calculate local statistics
    chi_squared = 0
    for s, vi, vj in itertools.product(range(5), repeat=3):
        observed = N_vi_vj_s[s, vi, vj]
        expected = N_vi_plus_s[s, vi] * N_plus_vj_s[s, vj] / N_plus_plus_s[s]
        chi_squared += (observed - expected) * (observed - expected) / expected

    d_result = cp.array(1, dtype=cp.double)
    kernel(
        (1,),
        (5,),
        (
            d_result,
            cp.asarray(N_vi_vj_s.flatten(), dtype=cp.uint32),
            cp.asarray(N_vi_plus_s.flatten(), dtype=cp.uint32),
            cp.asarray(N_plus_vj_s.flatten(), dtype=cp.uint32),
            cp.asarray(N_plus_plus_s.flatten(), dtype=cp.uint32),
            cp.int(dim_s),
            cp.int(dim_vi),
            cp.int(dim_vj),
        ),
    )

    print(f"expected - result {chi_squared - d_result}")
    assert abs(chi_squared - d_result) <= 0.1e-10
