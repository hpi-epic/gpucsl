from gpucsl.pc.helpers import (
    correlation_matrix_of,
    get_gaussian_thresholds,
    init_pc_graph,
)

from pathlib import Path
from typing import List, NamedTuple
import pytest
import networkx as nx
import pandas as pd
import numpy as np

from tests.fixtures.file_readers import read_pcalg_gml, read_sepsets_pcalg

alpha = 0.05
current_dir = Path(__file__).parent


"""
# This is how you can use multiple different fixtures as the same input for a testcase:

@pytest.mark.parametrize(
    "test_fixture",
    [
        pytest.lazy_fixture("input_data"),
        pytest.lazy_fixture("generated_input_data_and_results")
    ],
)
def test_fixture_example(test_fixture):

    expected_graph = test_fixture["expected_graph"]
"""

"""
# This is how you can specify multiple parameters to use for one fixture of a testcase:

@pytest.mark.parametrize("input_data", ["coolingData", "NCI-60"], indirect=True)
def test_fixture_example(input_data):
    pass
"""


class Fixture(NamedTuple):
    samples: np.ndarray
    fully_connected_graph: np.ndarray
    expected_graph: nx.DiGraph
    expected_graph_v_structures_only: nx.DiGraph
    sepsets: np.ndarray
    max_level_pcalg: int
    covariance_matrix: np.ndarray
    thresholds: List[float]
    dataset_name: str


def make_fixture(
    samples: np.ndarray,
    expected_graph: nx.DiGraph,
    expected_graph_v_structures_only: nx.DiGraph,
    sepsets: np.ndarray,
    max_level_pcalg: int,
    dataset_name: str,
) -> Fixture:
    fully_connected_graph = init_pc_graph(samples)
    thresholds = get_gaussian_thresholds(samples)
    covariance_matrix = correlation_matrix_of(samples)

    return Fixture(
        samples,
        fully_connected_graph,
        expected_graph,
        expected_graph_v_structures_only,
        sepsets,
        max_level_pcalg,
        covariance_matrix,
        thresholds,
        dataset_name,
    )


def create_input_fixture(dataset_name):
    data_folder = f"./data/{dataset_name}"
    data = pd.read_csv(f"{data_folder}/{dataset_name}.csv", header=None).to_numpy()

    (sepsets, max_level_pcalg) = read_sepsets_pcalg(
        f"{data_folder}/pcalg_{dataset_name}_sepset.txt", data.shape[1]
    )

    return make_fixture(
        data,
        read_pcalg_gml(f"{data_folder}/pcalg_{dataset_name}_graph.gml"),
        read_pcalg_gml(f"{data_folder}/pcalg_{dataset_name}_graph_v.gml"),
        sepsets,
        max_level_pcalg,
        dataset_name,
    )


@pytest.fixture(
    params=[
        "coolingData",
        "NCI-60",
        "MCC",
        "Saureus",
        "Scerevisiae",
        "DREAM5-Insilico",
        "BR51",
    ]
)
def input_data(request):
    return create_input_fixture(request.param)
