from typing import NamedTuple
import pandas as pd
import pytest
import networkx as nx
import numpy as np
from gpucsl.pc.helpers import init_pc_graph


class DiscreteFixture(NamedTuple):
    dataset_name: str
    initial_graph: np.ndarray
    data: np.ndarray
    expected_graph: nx.DiGraph
    memory_restriction: int
    max_level: int


ALPHA = 0.05


def read_input_data_discrete(dataset_name, max_level=20, memory_restriction=None):
    data = pd.read_csv(
        f"./data/{dataset_name}/{dataset_name}_encoded.csv", header=None
    ).to_numpy(dtype=np.uint8)
    initial_graph = init_pc_graph(data)
    expected_graph = nx.read_gml(
        path=f"./data/{dataset_name}/bnlearn_{dataset_name}_graph_max_level_{max_level}.gml",
        label="name",
    )
    return DiscreteFixture(
        dataset_name, initial_graph, data, expected_graph, memory_restriction, max_level
    )


test_configs = [
    *[("alarm", max_level, None) for max_level in [1, 3, 8, 11]],
    *[("link", max_level, None) for max_level in [1, 3]],
    *[("munin", max_level, None) for max_level in [1, 3]],
]


@pytest.fixture(params=test_configs)
def input_data_discrete(request):
    return read_input_data_discrete(
        request.param[0], request.param[1], request.param[2]
    )
