import numpy as np
import networkx as nx


def assert_pmax_valid(pmax: np.ndarray, gpucsl_result_graph: nx.DiGraph) -> bool:
    print("PMAX")
    print(pmax)

    skeleton = nx.to_numpy_array(gpucsl_result_graph.to_undirected())

    for i in range(skeleton.shape[0]):
        for j in range(skeleton.shape[1]):
            if i > j:
                assert pmax[i, j] == -1
            elif i == j:
                assert pmax[i, j] == -1
            else:
                if skeleton[i, j] == 1:
                    assert 0 <= pmax[i, j] and pmax[i, j] <= 1
                else:
                    assert 0 < pmax[i, j] and pmax[i, j] <= 1
