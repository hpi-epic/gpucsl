import pandas as pd
from pathlib import Path
import os
import numpy as np
import networkx as nx
from gpucsl.cli.cli_util import error
from gpucsl.pc.helpers import SkeletonResult


def force_csv_suffix(path: Path):
    if path.suffix != ".csv":
        path = path.parent / (path.stem + ".csv")

    return path


def get_file_name_stem(dataset_path: str):
    samples_path = Path(dataset_path)
    return samples_path.stem


def read_csv(path: Path):
    try:
        path = force_csv_suffix(Path(path))
        return pd.read_csv(path.absolute(), header=None).to_numpy()
    except FileNotFoundError as err:
        error(str(err))


def read_correlation_matrix(path: str):
    # if no path is given there is no precalculated matrix and it will be calculated
    # during the execution of the pc algorithm
    if path is None:
        return None

    return read_csv(path)


def write_seperation_set(output_path: str, file_name_stem: str, seperation_set):
    variable_count = seperation_set.shape[0]
    with open(output_path / (file_name_stem + "_sepset.txt"), "w", newline="") as f:
        # Nothing to write out here but create a file nevertheless to show there are no separation sets
        if seperation_set.size == 0:
            return

        for i in range(variable_count):
            for k in range(variable_count):
                s = seperation_set[i][k]
                if s[0] == -1:  # no separation set
                    continue
                set_items = s[np.where(s != -1)[0]]

                f.write(f"{i} {k} ")
                f.write(" ".join(map(str, set_items)))
                f.write("\n")


def write_directed_graph(output_path, file_name_stem, directed_graph):
    nx.write_gml(directed_graph, output_path / (file_name_stem + ".gml"))


def write_pmax(output_path, file_name_stem, pmax):
    np.savetxt(
        output_path / (file_name_stem + "_pmax.csv"),
        pmax,
        delimiter=",",
        fmt="%.4f",
    )


def write_configuration(output_path: str, file_name_stem: str, command_line_arguments):
    with open(output_path / (file_name_stem + "_config.txt"), "w", newline="") as f:
        f.write(
            f"The output for the dataset {file_name_stem} was generated using the following given parameters: \n\n"
        )
        f.write(" ".join(command_line_arguments))
        f.write("\n")


def write_results(output_path: str, file_name_stem: str, result: SkeletonResult):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    write_directed_graph(output_path, file_name_stem, result.directed_graph)
    write_seperation_set(output_path, file_name_stem, result.separation_sets)
    write_pmax(output_path, file_name_stem, result.pmax)

    print(f"successfully wrote results to: {output_path}")


def write_results_and_config(
    output_directory: str,
    dataset_path: str,
    command_line_arguments,
    result: SkeletonResult,
):
    output_path = Path(output_directory)
    file_name_stem = get_file_name_stem(dataset_path)

    write_results(output_path, file_name_stem, result)
    write_configuration(output_path, file_name_stem, command_line_arguments)
