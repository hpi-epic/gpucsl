from itertools import product
from pathlib import Path
from typing import NamedTuple
import pytest
import networkx as nx
import pandas as pd
import os
import numpy as np

from gpucsl.pc.pc import GaussianPC
from tests.fixtures.file_readers import (
    read_gpucsl_gml,
    read_ground_truth_gml,
    read_pcalg_gml,
    read_pmax,
    read_sepsets_pcalg,
)


alpha = 0.05
current_dir = Path(__file__).parent


class GeneratedResultsFixture(NamedTuple):
    dataset_name: str
    dataset_dir: str
    samples: np.ndarray
    ground_truth_graph: nx.DiGraph
    pcalg_skeleton: nx.Graph
    pcalg_graph: nx.DiGraph
    pcalg_sepsets: np.ndarray
    max_level_pcalg: int
    variable_count: int
    pcalg_pmax: np.ndarray
    gpucsl_skeleton: nx.Graph
    gpucsl_graph: nx.DiGraph
    gpucsl_pmax: np.ndarray


# try to use these configs for testing multiple configurations as this enables reuse (faster)
generated_test_data_configs = [
    f"--num_nodes {num_nodes} --edge_density {edge_density} --num_samples {num_samples} --discrete_node_ratio 0 --continuous_noise_std {noise}"
    for (num_nodes, edge_density, num_samples, noise) in product(
        [5, 10, 20],  # num_nodes
        [0.2, 0.3],  # edge_density
        [100000],  # num_samples
        [0.3, 0.4],  # noise
    )
]


def generate_input_data_and_results_if_needed(
    manm_cs_parameters: str, write_gpucsl=False
):
    dataset_name = "generated_" + manm_cs_parameters.replace("-", "").replace(" ", "_")
    dataset_dir = os.path.abspath(f"{current_dir}/../../data/{dataset_name}")
    print(current_dir)

    if not os.path.isdir(dataset_dir):
        print("Generating dataset " + dataset_dir)
        os.mkdir(dataset_dir)

    if not os.path.isfile(f"{dataset_dir}/ground_truth.gml"):
        generate_command = f'python3 -m manm_cs {manm_cs_parameters} --output_ground_truth_file "{dataset_dir}/ground_truth" --output_samples_file "{dataset_dir}/{dataset_name}"'
        print(generate_command)
        os.system(generate_command)

    if not os.path.isfile(f"{dataset_dir}/pcalg_{dataset_name}_graph.gml"):
        os.system(
            f"cd {current_dir}/../.. && ./scripts/use_pcalg_gaussian.R {dataset_name}"
        )
        os.system(f"cd {current_dir}")

    if not os.path.isfile(f"{dataset_dir}/.gitignore"):
        os.system(f'echo "*" > {dataset_dir}/.gitignore')

    if not os.path.isfile(f"{dataset_dir}/output.gml") and write_gpucsl:
        samples_path = f"{dataset_dir}/{dataset_name}.csv"
        data = pd.read_csv(samples_path, header=None).to_numpy()
        res = GaussianPC(data, 10).set_distribution_specific_options()
        nx.write_gml(res.result.directed_graph, f"{dataset_dir}/output.gml")

        if not os.path.isfile(f"{dataset_dir}/output_pmax.csv"):
            np.savetxt(
                f"{dataset_dir}/output_pmax.csv",
                res.result.pmax,
                delimiter=",",
                fmt="%.4f",
            )

    return (dataset_name, dataset_dir)


@pytest.fixture(
    params=generated_test_data_configs,
)
def generated_input_data_and_results(request):
    (dataset_name, dataset_dir) = generate_input_data_and_results_if_needed(
        request.param, True
    )

    data = pd.read_csv(f"{dataset_dir}/{dataset_name}.csv")
    data = data.reindex(sorted(data.columns), axis=1)
    data = data.to_numpy()

    (sepsets, max_level_pcalg) = read_sepsets_pcalg(
        f"{dataset_dir}/pcalg_{dataset_name}_sepset.txt", data.shape[1]
    )

    variable_count = data.shape[1]

    return GeneratedResultsFixture(
        dataset_name,
        dataset_dir,
        data,
        read_ground_truth_gml(f"{dataset_dir}/ground_truth.gml"),
        read_pcalg_gml(f"{dataset_dir}/pcalg_{dataset_name}_graph.gml").to_undirected(),
        read_pcalg_gml(f"{dataset_dir}/pcalg_{dataset_name}_graph.gml").to_directed(),
        sepsets,
        max_level_pcalg,
        variable_count,
        read_pmax(f"{dataset_dir}/pcalg_{dataset_name}_pMax.csv"),
        read_gpucsl_gml(f"{dataset_dir}/output.gml").to_undirected(),
        read_gpucsl_gml(f"{dataset_dir}/output.gml"),
        read_pmax(f"{dataset_dir}/output_pmax.csv"),
    )
