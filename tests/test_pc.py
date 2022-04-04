import numpy as np
from gpucsl.pc.pc import GaussianPC
from gpucsl.pc.helpers import correlation_matrix_of
from gpucsl.pc.kernel_management import Kernels
from tests.equality import check_graph_equality, graph_equality_isomorphic
from .fixtures.input_data import Fixture, input_data
import pytest
import networkx as nx
from .asserts import assert_pmax_valid
from typing import List

import cupy as cp
from .test_gaussian_device_manager import MockDevice


alpha = 0.05


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_pc(input_data: Fixture):
    data = input_data.samples
    expected_graph = input_data.expected_graph

    max_level = 3

    pc = GaussianPC(data, max_level, alpha).set_distribution_specific_options()
    (res, _) = pc.execute()

    assert check_graph_equality(
        expected_graph, res.directed_graph, graph_equality_isomorphic
    )

    assert_pmax_valid(res.pmax, res.directed_graph)

    print(nx.adjacency_matrix(res.directed_graph).todense())


@pytest.mark.timeout(3.0)
@pytest.mark.parametrize("devices", [[0], [0, 1]])
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_pc_interface(monkeypatch, input_data: Fixture, devices: List[int]):
    monkeypatch.setattr(cp.cuda, "Device", MockDevice)
    monkeypatch.setattr(cp.cuda.runtime, "getDeviceCount", lambda: 2)
    n_devices = len(devices)

    data = input_data.samples
    max_level = 3

    correlation_matrix = correlation_matrix_of(data)

    Kernels.for_gaussian_ci(data.shape[1], n_devices, max_level)

    kernels = [
        Kernels.for_gaussian_ci(data.shape[1], n_devices, max_level)
        for _ in range(n_devices)
    ]

    pc = GaussianPC(
        data,
        max_level,
        alpha,
        kernels=kernels,
    ).set_distribution_specific_options(
        correlation_matrix=correlation_matrix, devices=devices
    )

    result = pc.execute()

    res = result.result

    assert result.runtime > 0
    assert res.discover_skeleton_runtime > 0
    assert res.edge_orientation_runtime > 0
    assert res.directed_graph.is_directed()
    assert len(res.separation_sets) > 0
    assert res.pmax is not None
    assert res.discover_skeleton_kernel_runtime > 0


# Measures runtime. Only run this test manually with
# pytest tests/test_pc.py::test_pc_runtime_with_optional_args
# because if other tests run before it, kernels will already be compiled.
# Compare results with test_pc_runtime_without_supplied_kernels.
@pytest.mark.skip(reason="Only run this test manually (comment this line out)")
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_pc_runtime_with_supplied_kernels(input_data):
    data = input_data.samples
    max_level = 3

    correlation_matrix = correlation_matrix_of(data)
    kernels = [Kernels.for_gaussian_ci(data.shape[1], 1, max_level)]

    pc = GaussianPC(
        data,
        max_level,
        alpha,
        kernels,
    ).set_distribution_specific_options(correlation_matrix=correlation_matrix)

    (_, full_runtime) = pc.execute()

    print(f"pc duration: {full_runtime}")

    assert False


# Measures runtime. Only run this test manually with
# pytest tests/test_pc.py::test_pc_runtime_without_supplied_kernels
# because if other tests run before it, kernels will already be compiled.
# Compare results with test_pc_runtime_with_supplied_kernels.
@pytest.mark.skip(reason="Only run this test manually (comment this line out)")
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_pc_runtime_without_supplied_kernels(input_data):
    data = input_data["samples"]
    max_level = 3

    correlation_matrix = correlation_matrix_of(data)

    pc = GaussianPC(
        data,
        max_level,
        alpha,
    ).set_distribution_specific_options(correlation_matrix=correlation_matrix)

    (_, full_runtime) = pc.execute()
    print(f"pc duration: {full_runtime}")

    assert False
