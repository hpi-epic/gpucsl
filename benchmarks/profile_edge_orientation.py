import networkx as nx

from gpucsl.pc.edge_orientation.edge_orientation import orient_edges
from tests.fixtures import create_input_fixture

# separate script without pytest for easy profiling
if __name__ == "__main__":
    input_data = create_input_fixture("DREAM5-Insilico")

    skeleton = nx.Graph(input_data.expected_graph.to_undirected())

    (dag, _) = orient_edges(
        skeleton,
        input_data.sepsets,
    )
