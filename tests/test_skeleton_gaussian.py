import pytest
import networkx as nx
import numpy as np

from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian

from tests.equality import check_graph_equality, graph_equality_isomorphic
from .fixtures.input_data import Fixture, input_data
from .fixtures.generated_input_data_and_results import (
    GeneratedResultsFixture,
    generated_test_data_configs,
    generated_input_data_and_results,
)

alpha = 0.05


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
@pytest.mark.parametrize("is_debug", [True, False])
def test_gaussian_skeleton(input_data: Fixture, is_debug):

    initial_graph = input_data.fully_connected_graph
    expected_graph = input_data.expected_graph
    max_level = 3

    res = discover_skeleton_gpu_gaussian(
        initial_graph,
        input_data.samples,
        input_data.covariance_matrix,
        alpha,
        max_level,
        input_data.samples.shape[1],
        input_data.samples.shape[0],
        is_debug=is_debug,
    )
    skeleton = res.result.skeleton

    print(skeleton)

    skeleton_graph = nx.convert_matrix.from_numpy_array(skeleton)
    undirected_expected_graph = expected_graph.to_undirected()

    assert check_graph_equality(
        undirected_expected_graph, skeleton_graph, graph_equality_isomorphic
    )


@pytest.mark.parametrize(
    "generated_input_data_and_results",
    generated_test_data_configs,
    indirect=True,
)
@pytest.mark.run_slow
def test_gaussian_skeleton_generated_pmax(
    generated_input_data_and_results: GeneratedResultsFixture,
):
    d = generated_input_data_and_results

    print(d.pcalg_pmax)
    print(d.gpucsl_pmax)
    pmax_is_close = np.isclose(d.pcalg_pmax, d.gpucsl_pmax, atol=0.01)
    print(pmax_is_close)
    percentage_pmax_is_close = np.sum(pmax_is_close) / np.size(pmax_is_close)
    print(percentage_pmax_is_close)
    assert percentage_pmax_is_close > 0.65


@pytest.mark.parametrize(
    "generated_input_data_and_results",
    generated_test_data_configs,
    indirect=True,
)
@pytest.mark.run_slow
def test_gaussian_skeleton_generated_adjacency(
    generated_input_data_and_results: GeneratedResultsFixture,
):

    d = generated_input_data_and_results

    pcalg_adjacency = nx.to_numpy_array(d.pcalg_skeleton)
    gpucsl_adjacency = nx.to_numpy_array(d.gpucsl_skeleton)

    print("PCALG ADJENCENCY")
    print(pcalg_adjacency)
    print("GPUCSL ADJACENCY")
    print(gpucsl_adjacency)

    adjacency_is_close = np.isclose(pcalg_adjacency, gpucsl_adjacency)
    print(adjacency_is_close)
    percentage_is_close = np.sum(adjacency_is_close) / np.size(adjacency_is_close)
    assert percentage_is_close == 1.0
