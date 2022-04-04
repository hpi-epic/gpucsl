from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from gpucsl.pc.edge_orientation.orientation_rules import apply_rules
from gpucsl.pc.edge_orientation.v_structures import orient_v_structure
from gpucsl.pc.helpers import timed

# Takes the undirected graph, represented as directed graph with edges in both
# directions, and directs it by removing edges that can be oriented due to
# v_structures or orientation rules
# seperation_sets is expected to have both (v_1, v_2) and (v_2, v_1) as a valid key
# to the seperation set
#
# Orientation rules are implemented according to
# Christopher Meek. 1995. Causal inference and causal explanation with background
# knowledge. In Proceedings of the Eleventh conference on Uncertainty in artificial
# intelligence (UAI'95)
# Find example ascii-graphs in ../tests/test_edge_orientation.py

# Parameters
# - skeleton: nx.Graph, undirected
# - separation_sets: np.ndarray of shape (variable_count, variable_count, max_level)
@timed
def orient_edges(
    skeleton: nx.Graph,
    separation_sets: np.ndarray,
) -> nx.DiGraph:
    dag = skeleton.to_directed()

    orient_v_structure(dag, separation_sets, skeleton)

    apply_rules(dag)

    return dag
