from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def sepset_dict_to_ndarray(
    sepsets: Dict[Tuple[int, int], Set[int]], variable_count: int, max_level: int
) -> np.ndarray:
    separation_sets = np.full((variable_count, variable_count, max_level), -1, np.int32)

    for (v_i, v_j), value in sepsets.items():
        sepset_list = list(value)
        sepset_list.extend([-1] * (max_level - len(sepset_list)))
        separation_sets[v_i][v_j] = sepset_list

    return separation_sets


# experimentally determined for the current datasets, will throw if exceeded
MAX_LEVEL_SEPSETS_PCALG = 20


def read_sepsets_pcalg(file_path: str, variable_count: int) -> np.ndarray:

    separation_sets = np.full(
        (variable_count, variable_count, MAX_LEVEL_SEPSETS_PCALG), -1, np.int32
    )

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            v_1, v_2, sepset_length, *sepset_list = line.split()
            assert int(sepset_length) == len(sepset_list)

            # pcalg sepsets use the label of the nodes. Transform them to use zero-indexed ints.
            v_1 = int(v_1) - 1
            v_2 = int(v_2) - 1
            sepset_list = [int(v) - 1 for v in list(sepset_list)]

            sepset_list_dedup = list(set(sepset_list))
            sepset_list_dedup.extend(
                [-1] * (MAX_LEVEL_SEPSETS_PCALG - len(sepset_list_dedup))
            )
            separation_sets[v_1][v_2] = sepset_list_dedup
            separation_sets[v_2][v_1] = sepset_list_dedup

    return (separation_sets, MAX_LEVEL_SEPSETS_PCALG)


# Convention: for all graphs use id as label, discard all other attributes
# so that equality checks are easier
def read_gml(
    file_path: str, label_attr: str, attributes_to_remove: List[str]
) -> nx.DiGraph:
    graph = nx.read_gml(path=file_path, label=label_attr)
    for _, attr in graph.nodes(data=True):
        for attribute_name in attributes_to_remove:
            del attr[attribute_name]

    return graph


def read_pcalg_gml(file_path):
    return read_gml(file_path, "id", ["name"])


def read_ground_truth_gml(file_path):
    return read_gml(file_path, "id", ["label"])


def read_gpucsl_gml(file_path: str):
    return read_gml(file_path, "id", ["label"])


def read_pmax(file_path: str) -> np.ndarray:
    return pd.read_csv(file_path, sep=",", header=None).to_numpy()
