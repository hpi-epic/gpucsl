import networkx as nx
import pytest

from gpucsl.pc.edge_orientation.v_structures import orient_v_structure
from tests.equality import check_graph_equality, graph_equality_edges
from tests.fixtures.file_readers import read_pcalg_gml, sepset_dict_to_ndarray
from tests.fixtures.input_data import input_data


@pytest.fixture(params=[orient_v_structure])
def orient_v_input_simple(request):
    set_with_2 = {2}
    separation_sets = sepset_dict_to_ndarray(
        {(1, 3): set_with_2, (3, 1): set_with_2}, 4, 2
    )

    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3])

    edges = [(1, 2), (2, 1), (2, 3), (3, 2)]
    g.add_edges_from(edges)

    sepsets_empty = sepset_dict_to_ndarray({}, 4, 2)

    return g, separation_sets, sepsets_empty, edges, request.param


# Test
# ┌─────────────────┐
# │v_1 -- v_2 -- v_3│
# └─────────────────┘
#          to
# ┌─────────────────┐
# │v_1 -> v_2 <- v_3│
# └─────────────────┘
# iff v_1 and v_3 are non adjacent and v_2 not in Seperation set of v_1 and v_3
def test_orient_v_structure_simple(orient_v_input_simple):

    (g, separation_sets, sepsets_empty, edges, orient_func) = orient_v_input_simple

    orient_func(g, separation_sets)
    assert all([g.has_edge(*edge) for edge in edges])

    orient_func(g, sepsets_empty)
    assert not g.has_edge(2, 1) and not g.has_edge(2, 3)
    assert g.has_edge(1, 2) and g.has_edge(3, 2)


@pytest.fixture(params=[orient_v_structure])
def orient_v_input_pc_example(request):

    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3, 4])

    edges = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
    g.add_edges_from(edges)

    sepsets_empty = sepset_dict_to_ndarray({}, 5, 2)

    return g, sepsets_empty, edges, request.param


# Test
# ┌────────────────────────┐
# │v_1 -- v_2 -- v_3 -- v_4│
# └────────────────────────┘
#          to
# ┌────────────────────────┐
# │v_1 -> v_2 -> v_3 <- v_4│
# └────────────────────────┘
# resolve conflicting triples according to pcalg documentation without solve conflict
def test_orient_v_structure_pc_example(orient_v_input_pc_example):

    (g, sepsets_empty, edges, orient_func) = orient_v_input_pc_example

    orient_func(g, sepsets_empty)
    assert not g.has_edge(2, 1) and not g.has_edge(3, 2) and not g.has_edge(3, 4)
    assert g.has_edge(1, 2) and g.has_edge(2, 3) and g.has_edge(4, 3)


@pytest.fixture(params=[orient_v_structure])
def orient_v_input_partially_directed(request):
    set_with_2 = {2}
    separation_sets = sepset_dict_to_ndarray(
        {(1, 3): set_with_2, (3, 1): set_with_2}, 4, 2
    )

    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3])

    edges = [(1, 2), (2, 1), (3, 2)]
    g.add_edges_from(edges)

    sepsets_empty = sepset_dict_to_ndarray({}, 4, 2)

    return g, separation_sets, sepsets_empty, edges, request.param


def test_orient_v_structure_already_partially_directed(
    orient_v_input_partially_directed,
):
    (
        g,
        separation_sets,
        sepsets_empty,
        edges,
        orient_func,
    ) = orient_v_input_partially_directed

    orient_func(g, separation_sets)
    assert all([g.has_edge(*edge) for edge in edges])

    orient_func(g, sepsets_empty)
    assert not g.has_edge(2, 1) and not g.has_edge(2, 3)
    assert g.has_edge(1, 2)


def run_orient_v_structure_pcalg_comparison(i):
    graph_v = i.expected_graph_v_structures_only.to_directed()
    skeleton = i.expected_graph.to_undirected()
    gpucsl_dag = skeleton.to_directed()

    orient_v_structure(
        gpucsl_dag,
        i.sepsets,
        skeleton,
    )

    assert check_graph_equality(graph_v, gpucsl_dag, graph_equality_edges)


@pytest.mark.parametrize(
    "input_data",
    [
        "coolingData",
    ],
    indirect=True,
)
def test_orient_v_structures_compare_pcalg(
    input_data,
):
    run_orient_v_structure_pcalg_comparison(input_data)


@pytest.mark.run_slow
def test_orient_v_structures_compare_pcalg(
    input_data,
):
    run_orient_v_structure_pcalg_comparison(input_data)
