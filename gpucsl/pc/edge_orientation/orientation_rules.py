import itertools
import logging

import networkx as nx


def apply_rules(dag: nx.DiGraph):
    def non_adjacent(g, v_1, v_2):
        return not g.has_edge(v_1, v_2) and not g.has_edge(v_2, v_1)

    def existing_edge_is_directed_only(g, v_1, v_2):
        return not g.has_edge(v_2, v_1)

    def undirected(g, v_1, v_2):
        return g.has_edge(v_1, v_2) and g.has_edge(v_2, v_1)

    num_nodes = len(dag.nodes)

    def column_major_edge_ordering(edge):
        return edge[1] * num_nodes + edge[0]

    while True:
        graph_changed = False

        # Rule 1
        # v_1 -> v_2 - v_3 to v_1 -> v_2 -> v_3
        dag2 = dag.copy()
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if dag2.has_edge(v_2, v_1):
                continue
            for v_3 in sorted(dag2.successors(v_2)):
                if v_1 == v_3:
                    continue
                if dag2.has_edge(v_3, v_2) and non_adjacent(dag2, v_1, v_3):
                    # only no conflict solution
                    if undirected(dag, v_2, v_3):
                        logging.debug(f"R1: remove ({v_3, v_2})")
                        dag.add_edge(v_2, v_3)
                        dag.remove_edges_from([(v_3, v_2)])
                        graph_changed = True

        # Rule 2
        # v_1 -> v_3 -> v_2 with v_1 - v_2: v_1 -> v_2
        dag2 = dag.copy()  # work on current dag after Rule 1
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if not dag2.has_edge(v_2, v_1):
                continue
            for v_3 in sorted(
                set(dag2.successors(v_1)).intersection(dag2.predecessors(v_2))
            ):
                if existing_edge_is_directed_only(
                    dag2, v_1, v_3
                ) and existing_edge_is_directed_only(dag2, v_3, v_2):
                    logging.debug(f"R2: remove ({v_2, v_1})")
                    dag.add_edge(v_1, v_2)
                    dag.remove_edges_from([(v_2, v_1)])
                    graph_changed = True

        # Rule 3
        # ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
        # │v_3├───┤v_1├───┤v_4│   │v_3├───┤v_1├───┤v_4│
        # └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
        #   │       │       │  to   │       │       │
        #   │       │       │       │       │       │
        #   │     ┌─┴─┐     │       │     ┌─▼─┐     │
        #   └────►│v_2│◄────┘       └────►│v_2│◄────┘
        #         └───┘                   └───┘
        dag2 = dag.copy()  # work on current dag after Rule 2
        for v_1, v_2 in sorted(dag2.edges, key=column_major_edge_ordering):
            if not dag2.has_edge(v_2, v_1):
                continue
            neighbors_v1 = set(dag2.successors(v_1)).intersection(
                dag2.predecessors(v_1)
            )
            predecessors_v2 = set(dag2.predecessors(v_2)).difference(
                dag2.successors(v_2)
            )
            C = sorted(
                neighbors_v1.intersection(predecessors_v2),
            )
            for v_3, v_4 in itertools.combinations(C, 2):
                if non_adjacent(dag2, v_3, v_4):
                    logging.debug(f"R3: remove ({v_2, v_1})")
                    dag.add_edge(v_1, v_2)
                    dag.remove_edges_from([(v_2, v_1)])
                    graph_changed = True

        if not graph_changed:
            return
