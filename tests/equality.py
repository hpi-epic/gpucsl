from typing import Callable
import networkx as nx
import numpy as np


def graph_equality_isomorphic(expected_graph, actual_graph):
    return nx.is_isomorphic(
        expected_graph, actual_graph, node_match=lambda x, y: x == y
    )


def graph_equality_edges(expected_graph, actual_graph):
    return set(actual_graph.edges()) == set(expected_graph.edges())


def check_graph_equality(expected_graph, actual_graph, equality: Callable):
    print(expected_graph)
    print(actual_graph)

    actual_edges = set(actual_graph.edges())
    expected_edges = set(expected_graph.edges())

    expected_length = len(expected_graph.edges)
    actual_length = len(actual_graph.edges)

    if expected_length < actual_length:
        print(f"Actual has {actual_length - expected_length} edges too many")

    if expected_length > actual_length:
        print(f"Expected has {expected_length - actual_length} more edges")

    print(f"Correct edges: {actual_edges.intersection(expected_edges)}\n")
    print(
        f"Edges that are in expected but not in actual are: {expected_edges.difference(actual_edges)}\n"
    )
    print(
        f"Edges that are in actual but not in expected are: {actual_edges.difference(expected_edges)}\n"
    )

    print("calculated graph: ")
    print(nx.adjacency_matrix(actual_graph).todense())
    print("\nexpected graph: ")
    print(nx.adjacency_matrix(expected_graph).todense())
    print("\ndifference:")
    print(
        np.isclose(
            nx.to_numpy_array(actual_graph), nx.to_numpy_array(expected_graph)
        ).astype(int)
    )

    return equality(expected_graph, actual_graph)
