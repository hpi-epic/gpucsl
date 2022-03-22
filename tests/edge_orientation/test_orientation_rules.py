import networkx as nx

from gpucsl.pc.edge_orientation.orientation_rules import apply_rules


# Rule 1
# ┌─────────────────┐
# │v_1 -- v_2 -> v_3│
# └─────────────────┘
#          to
# ┌─────────────────┐
# │v_1 -> v_2 -> v_3│
# └─────────────────┘
# Orient V_1 - V_2 to V_1 --> V_2 whenever there is an arrow V_2 --> V_3
# Otherwise, this would introduce a new v-structure
def test_rule_one_simple():
    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3])
    g.add_edges_from([(1, 2), (2, 3), (3, 2)])

    apply_rules(g)

    assert not g.has_edge(3, 2)
    assert g.has_edge(1, 2) and g.has_edge(1, 2)


# Rule 2
# ┌───┐     ┌───┐
# │v_1│     │v_1│
# └┬─┬┘     └┬─┬┘
#  │┌▽──┐    │┌▽──┐
#  ││v_3│ to ││v_3│
#  │└┬──┘    │└┬──┘
# ┌┴─▽┐     ┌▽─▽┐
# │v_2│     │v_2│
# └───┘     └───┘
# Orient V_1 - V_2 to V_1 --> V_2 whenever there is a chain V_1 --> V_3 --> V_2
# Otherwise, we get a cycle
def test_rule_two_simple():
    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3])
    g.add_edges_from([(1, 3), (3, 1), (1, 2), (2, 3)])

    apply_rules(g)

    assert not g.has_edge(3, 1)
    assert g.has_edge(1, 3) and g.has_edge(1, 2) and g.has_edge(2, 3)


# Rule 3
# ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
# │v_3├───┤v_1├───┤v_4│   │v_3├───┤v_1├───┤v_4│
# └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
#   │       │       │  to   │       │       │
#   │       │       │       │       │       │
#   │     ┌─┴─┐     │       │     ┌─▼─┐     │
#   └────►│v_2│◄────┘       └────►│v_2│◄────┘
#         └───┘                   └───┘
# Orient V_1 - V_2 to V_1 --> V_2 whenever there are two chaing V_1 - V_3 --> V2,
# V_1 - V_4 --> V_2  and V_3 and V_4 are nonadjacent
def test_rule_three_simple():
    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3, 4])
    g.add_edges_from([(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (3, 2), (4, 2)])

    apply_rules(g)

    assert not g.has_edge(2, 1)

    # test if all other edges are present
    for edge in [(1, 2), (1, 3), (3, 1), (1, 4), (4, 1), (3, 2), (4, 2)]:
        assert g.has_edge(*edge)
