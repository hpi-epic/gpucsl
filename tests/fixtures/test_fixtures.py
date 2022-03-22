from .input_data import input_data
from .generated_input_data_and_results import generated_input_data_and_results

import pytest
import networkx as nx
import pandas as pd
from gpucsl import pc


@pytest.mark.run_slow
def test_generated_input_data_and_results(generated_input_data_and_results):
    pass


@pytest.mark.run_slow
def test_input_data(input_data):
    pass


@pytest.mark.parametrize(
    "input_data",
    ["coolingData"],
    indirect=True,
)
def test_input_data(input_data):
    pass


@pytest.mark.parametrize(
    "generated_input_data_and_results",
    [
        "--num_nodes 5 --edge_density 0.35 --num_samples 100000 --discrete_node_ratio 0 --continuous_noise_std 0.3"
    ],
    indirect=True,
)
def test_generated_input_data_and_results_one_dataset(generated_input_data_and_results):
    # always run for one folder, only run for all in run_slow config
    dataset_name = generated_input_data_and_results.dataset_name
    print(dataset_name)
    assert True
