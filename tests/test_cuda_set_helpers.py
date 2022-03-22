import cupy as cp
import numpy as np
import pandas as pd
import itertools
import math
import networkx as nx
from gpucsl.pc.kernel_management import get_module

function_names = [
    "calculate_sepset_indices<3>",
]
module = get_module("helpers/set_helpers.cu", function_names, ("-D", "PYTHON_TEST"))


def test_calculate_sepset_indices():
    level = 3
    kernel = module.get_function("calculate_sepset_indices<3>")

    d_neighbours_count = cp.int32(5)

    expected_result = np.array(
        [
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 5],
            [1, 3, 4],
            [1, 3, 5],
            [1, 4, 5],
            [2, 3, 4],
            [2, 3, 5],
            [2, 4, 5],
            [3, 4, 5],
        ]
    )

    n_sepsets = 10  # BinomialCoefficient(d_neighbours_count, level)
    d_result = cp.zeros((level,), np.int32)

    for sepset_index in range(n_sepsets):
        d_sepset_index = cp.int32(sepset_index)
        kernel((1, 1), (1, 1), (d_neighbours_count, d_sepset_index, d_result))
        assert cp.isclose(expected_result[sepset_index], d_result).all()
