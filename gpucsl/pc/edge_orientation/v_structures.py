import itertools
import logging

import networkx as nx
import numpy as np


def orient_v_structure(
    dag: nx.DiGraph,
    separation_sets: np.ndarray,
    skeleton: nx.Graph = None,
) -> None:
    def in_separation_set(v, v_1, v_2):
        return v in separation_sets[v_1][v_2]

    if skeleton is None:
        skeleton = dag.to_undirected()

    def non_adjacent(v_1, v_2):
        return not skeleton.has_edge(v_1, v_2)

    num_nodes = len(skeleton.nodes)

    for v_1, v_2 in sorted(
        skeleton.to_directed().edges, key=lambda x: x[1] * num_nodes + x[0]
    ):
        for v_3 in sorted(skeleton.neighbors(v_2), reverse=False):
            if v_1 == v_3:
                continue
            if non_adjacent(v_1, v_3) and not (
                in_separation_set(v_2, v_1, v_3) or in_separation_set(v_2, v_3, v_1)
            ):
                logging.debug(f"v: {[(v_2, v_1), (v_2, v_3)]}")
                dag.add_edges_from([(v_1, v_2), (v_3, v_2)])
                dag.remove_edges_from([(v_2, v_1), (v_2, v_3)])
