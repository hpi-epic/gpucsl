import cupy as cp
import numpy as np
import pandas as pd
import itertools
import math
import networkx as nx
from gpucsl.pc.kernel_management import get_module

function_names = [
    "compute_matrix_multiplication<2,3,2>",
    "compute_matrix_multiplication_with_first_transposed<3,2,3>",
    "compute_matrix_multiplication_with_second_transposed<2,3,2>",
    "compute_inverse_matrix<3>",
    "cholesky_decomposition<3>",
    "compute_m_2_inv<3>",
]
module = get_module(
    "helpers/gaussian_helpers.cu", function_names, ("-D", "PYTHON_TEST")
)


def test_matrix_multiplication():
    kernel = module.get_function("compute_matrix_multiplication<2,3,2>")

    a = np.arange(1, 7).reshape((2, 3))
    b = np.arange(1, 7).reshape((3, 2))
    expected_result = np.matmul(a, b)

    d_a = cp.asarray(a, cp.double)
    d_b = cp.asarray(b, cp.double)
    d_result = cp.zeros((2, 2), np.double)

    kernel((1, 1), (1, 1), (d_a, d_b, d_result))

    assert cp.isclose(expected_result, d_result).all()


def test_matrix_multiplication_transpose_first():
    kernel = module.get_function(
        "compute_matrix_multiplication_with_first_transposed<3,2,3>"
    )

    a = np.arange(1, 7).reshape((2, 3))
    b = np.arange(1, 7).reshape((2, 3))
    a_transposed = np.transpose(a)
    expected_result = np.matmul(a_transposed, b)

    d_a = cp.asarray(a, cp.double)
    d_b = cp.asarray(b, cp.double)
    d_result = cp.zeros((3, 3), np.double)

    kernel((1, 1), (1, 1), (d_a, d_b, d_result))

    assert cp.isclose(expected_result, d_result).all()


def test_matrix_multiplication_transpose_second():
    kernel = module.get_function(
        "compute_matrix_multiplication_with_second_transposed<2,3,2>"
    )

    a = np.arange(1, 7).reshape((2, 3))
    b = np.arange(1, 7).reshape((2, 3))
    b_transposed = np.transpose(b)
    expected_result = np.matmul(a, b_transposed)

    d_a = cp.asarray(a, cp.double)
    d_b = cp.asarray(b, cp.double)
    d_result = cp.zeros((2, 2), np.double)

    kernel((1, 1), (1, 1), (d_a, d_b, d_result))

    assert cp.isclose(expected_result, d_result).all()


def test_compute_inverse_of_square_matrix():
    kernel = module.get_function("compute_inverse_matrix<3>")

    # some random invertable(!) matrix
    a = np.array([[5, 3, 4], [3, 4, 5], [6, 3, 2]])
    expected_result = np.linalg.inv(a)

    d_a = cp.asarray(a, cp.double)
    d_result = cp.zeros((3, 3), np.double)

    kernel((1, 1), (1, 1), (d_a, d_result, cp.int_(3)))

    assert cp.isclose(expected_result, d_result).all()


def test_cholesky_decomposition():
    kernel = module.get_function("cholesky_decomposition<3>")

    # some random positive definite matrix
    a = np.array(
        [[4.2, 12.3, -16.4], [12.3, 37.3, -43.3], [-16.4, -43.5, 98.6]], np.double
    )
    expected_result = np.linalg.cholesky(a)

    d_a = cp.asarray(a, cp.double)
    d_result = cp.zeros((3, 3), np.double)

    kernel((1, 1), (1, 1), (d_a, d_result, cp.int_(3)))

    assert cp.isclose(expected_result, d_result).all()


def test_compute_M_2_inv():
    kernel = module.get_function("compute_m_2_inv<3>")

    a = np.array(
        [
            [1, 2, 3],
            [5, 2, 2],
            [1, 6, 1],
        ],
        dtype=np.double,
    )

    expectet_result = np.linalg.pinv(a)

    d_a = cp.asarray(a, cp.double)
    d_result = cp.zeros((3, 3), np.double)

    kernel((1, 1), (1, 1), (d_a, d_result))

    assert cp.isclose(expectet_result, d_result).all()
