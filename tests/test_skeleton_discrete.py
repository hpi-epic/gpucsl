import pytest
import networkx as nx
import cupy as cp

from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from tests.equality import check_graph_equality, graph_equality_isomorphic

from .fixtures.input_data_discrete import DiscreteFixture, input_data_discrete

from .asserts import assert_pmax_valid

ALPHA = 0.05

# =========== HELPERS ===========


def check_discrete_result(result, expected_graph):

    skeleton = result.skeleton
    n_variables = skeleton.shape[0]
    result_graph = nx.convert_matrix.from_numpy_array(skeleton)
    undirected_expected_graph = expected_graph.to_undirected()
    mapping = {
        ("V" + str(i + 1)): i for i in range(n_variables)
    }  # execute relabeling in order to compare the graphs
    relabed_undirected_expected_graph = nx.relabel_nodes(
        undirected_expected_graph, mapping
    )

    print(skeleton)
    print(f"skeleton has {len(result_graph.edges())} edges")
    print(f"expected has {len(undirected_expected_graph.edges())} edges")
    print(
        f"edges missing in result: {set(relabed_undirected_expected_graph.edges()) - set(result_graph.edges())}"
    )
    print(
        f"edges that are in result but should not be: {set(result_graph.edges()) - set(relabed_undirected_expected_graph.edges())}"
    )
    assert check_graph_equality(
        relabed_undirected_expected_graph, result_graph, graph_equality_isomorphic
    )
    assert_pmax_valid(result.pmax, nx.DiGraph(result.skeleton))


def run_discrete_skeleton_with_max_level(
    initial_graph, data, expected_graph, max_level, is_debug
):

    result, _ = discover_skeleton_gpu_discrete(
        initial_graph,
        data,
        ALPHA,
        max_level,
        data.shape[1],
        data.shape[0],
        is_debug=is_debug,
    )
    check_discrete_result(result, expected_graph)


### =========== TESTS ===========


@pytest.mark.parametrize(
    "input_data_discrete",
    [
        *[("alarm", max_level, None) for max_level in [1, 3, 8, 11]],
        *[("link", max_level, None) for max_level in [1, 3]],
        # *[("munin", max_level, None) for max_level in [1, 3]], # > 3 min exec time
    ],
    indirect=True,
)
@pytest.mark.run_slow
def test_discrete_skeleton_all_datasets(input_data_discrete: DiscreteFixture):
    i = input_data_discrete
    run_discrete_skeleton_with_max_level(
        i.initial_graph, i.data, i.expected_graph, i.max_level, False
    )


@pytest.mark.parametrize(
    "input_data_discrete",
    [("alarm", max_level, None) for max_level in [1, 11]],
    indirect=True,
)
def test_discrete_skeleton_max_level(input_data_discrete: DiscreteFixture):
    i = input_data_discrete
    run_discrete_skeleton_with_max_level(
        i.initial_graph, i.data, i.expected_graph, i.max_level, True
    )


# This test is to make sure that the cleanup logic works
# It just runs the first test two times
@pytest.mark.parametrize("input_data_discrete", [("alarm", 1, None)], indirect=True)
def test_discrete_skeleton_level_1_two_times_in_a_row(
    input_data_discrete: DiscreteFixture,
):
    i = input_data_discrete

    # One
    run_discrete_skeleton_with_max_level(
        i.initial_graph.copy(), i.data.copy(), i.expected_graph.copy(), 1, False
    )
    # Two
    run_discrete_skeleton_with_max_level(
        i.initial_graph.copy(), i.data.copy(), i.expected_graph.copy(), 1, False
    )


# The memory restrictions are chosen so that a high memory_reduction_factor is necessary,
# and it can be checked if the kernel still produces correct outputs
@pytest.mark.parametrize(
    "input_data_discrete",
    [
        ("alarm", max_level, restr)
        for (max_level, restr) in [(1, 100_000), (3, 500_000)]
    ],
    indirect=True,
)
@pytest.mark.parametrize("is_debug", [True, False])
def test_discrete_skeleton_with_memory_reduction(
    input_data_discrete: DiscreteFixture, is_debug
):
    i = input_data_discrete

    result, _ = discover_skeleton_gpu_discrete(
        i.initial_graph,
        i.data,
        ALPHA,
        i.max_level,
        i.data.shape[1],
        i.data.shape[0],
        memory_restriction=i.memory_restriction,
        is_debug=is_debug,
    )
    check_discrete_result(result, i.expected_graph)


@pytest.mark.parametrize(
    "input_data_discrete",
    [("alarm", 8, None)],
    indirect=True,
)
@pytest.mark.parametrize("is_debug", [True, False])
def test_discrete_skeleton_with_blocked_memory(
    input_data_discrete: DiscreteFixture, is_debug
):
    i = input_data_discrete

    free_memory, total_memory = cp.cuda.runtime.memGetInfo()

    # In full parallelization, level 8 needs around 9GB of space
    # This is why we reduce it to 5GB of free space, to test if
    # If the test still works, we know that the algorithm detected the shortage and used less memory
    five_gb = 5_000_000_000

    assert free_memory > five_gb

    # allocate memory to reduce free memory
    _ = cp.cuda.Memory(free_memory - five_gb)

    result, _ = discover_skeleton_gpu_discrete(
        i.initial_graph,
        i.data,
        ALPHA,
        i.max_level,
        i.data.shape[1],
        i.data.shape[0],
        is_debug=is_debug,
    )
    check_discrete_result(result, i.expected_graph)
