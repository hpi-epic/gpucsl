import cupy as cp
import numpy as np
import pandas as pd
import itertools
import math
import networkx as nx
from gpucsl.pc.kernel_management import get_module

function_names = [
    "compact<6,6>",
]
module = get_module("helpers/graph_helpers.cu", function_names, ("-D", "PYTHON_TEST"))


def test_compact_on_random_skeleton():
    kernel = module.get_function("compact<6,6>")

    d_skeleton = cp.array(
        [
            [0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 1, 0, 0],
        ],
        np.uint16,
    )

    expected_result = np.array(
        [
            [3, 1, 2, 5, 0, 0],
            [3, 0, 2, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [5, 0, 1, 2, 4, 5],
            [3, 0, 2, 5, 0, 0],
            [3, 1, 2, 3, 0, 0],
        ],
        np.uint32,
    )

    d_compacted_skeleton = cp.zeros((6, 6), np.uint32)

    kernel((1,), (6,), (d_skeleton, d_compacted_skeleton, 0, 6))

    assert cp.isclose(expected_result, d_compacted_skeleton).all()


def test_compact_on_fully_connected_skeleton():
    kernel = module.get_function("compact<6,6>")

    d_skeleton = cp.ones((6, 6), np.uint16)

    expected_result = np.array(
        [
            [5, 1, 2, 3, 4, 5],
            [5, 0, 2, 3, 4, 5],
            [5, 0, 1, 3, 4, 5],
            [5, 0, 1, 2, 4, 5],
            [5, 0, 1, 2, 3, 5],
            [5, 0, 1, 2, 3, 4],
        ],
        np.uint32,
    )

    d_compacted_skeleton = cp.zeros((6, 6), np.uint32)

    kernel((1,), (6,), (d_skeleton, d_compacted_skeleton, 0, 6))

    assert cp.array_equal(expected_result, d_compacted_skeleton.get())


def test_compact_on_random_big_skeleton():
    kernel = module.get_function("compact<6,6>")

    size = 5000

    d_skeleton = cp.random.choice([0, 1], size=(size, size)).astype(np.uint16)

    d_compacted_skeleton = cp.zeros((6, 6), np.uint32)

    print((math.ceil(size / 512),))
    print((min(512, size),))

    cp.cuda.profiler.start()
    kernel(
        (math.ceil(size / 512),),
        (min(512, size),),
        (d_skeleton, d_compacted_skeleton, 0, size),
    )
    cp.cuda.profiler.stop()
