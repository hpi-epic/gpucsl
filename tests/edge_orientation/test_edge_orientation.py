from itertools import product
import networkx as nx
import pytest

from gpucsl.pc.edge_orientation.edge_orientation import orient_edges
from tests.equality import (
    check_graph_equality,
    graph_equality_edges,
    graph_equality_isomorphic,
)
from tests.fixtures.generated_input_data_and_results import (
    GeneratedResultsFixture,
    generated_input_data_and_results,
    generated_test_data_configs,
)
from tests.fixtures.input_data import Fixture, input_data


def run_edge_orientation_compare_pcalg(input_data: Fixture):
    # Create undirected version of the expected graph (which is the skeleton)
    skeleton = input_data.expected_graph.to_undirected()

    print(input_data.sepsets)
    (dag, runtime) = orient_edges(
        skeleton,
        input_data.sepsets,
    )

    print(runtime)

    # isomorphic equality freezes for Scerevisiae dataset
    assert check_graph_equality(input_data.expected_graph, dag, graph_equality_edges)


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_edge_orientation_compare_pcalg(input_data: Fixture):
    run_edge_orientation_compare_pcalg(input_data)


@pytest.mark.run_slow
def test_edge_orientation_compare_pcalg_all(input_data: Fixture):
    run_edge_orientation_compare_pcalg(input_data)


@pytest.mark.parametrize(
    "generated_input_data_and_results",
    generated_test_data_configs,
    indirect=True,
)
@pytest.mark.run_slow
def test_edge_orientation_generated_data(
    generated_input_data_and_results: GeneratedResultsFixture,
):
    d = generated_input_data_and_results

    (dag, runtime) = orient_edges(
        d.pcalg_skeleton,
        d.pcalg_sepsets,
    )

    print(runtime)

    assert check_graph_equality(d.pcalg_graph, dag, graph_equality_isomorphic)
