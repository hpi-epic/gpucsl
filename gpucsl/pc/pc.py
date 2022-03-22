from enum import Enum, auto
from typing import List

import networkx as nx
import numpy as np
from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from gpucsl.pc.edge_orientation.edge_orientation import orient_edges
from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian

from gpucsl.pc.helpers import PCResult, init_pc_graph, timed
from gpucsl.pc.kernel_management import KernelGetter


class DataDistribution(Enum):
    GAUSSIAN = auto()
    DISCRETE = auto()


@timed
def pc(
    data: np.ndarray,
    data_distribution: DataDistribution,
    max_level: int,
    alpha=0.05,
    kernels=None,
    is_debug: bool = False,
    should_log: bool = False,
    devices: List[int] = [0],
    sync_device: int = None,
    gaussian_correlation_matrix: np.ndarray = None,
    discrete_max_memory_size=None,
) -> PCResult:
    num_variables = data.shape[1]
    num_observations = data.shape[0]

    assert len(devices) > 0, "Invalid device list, must not be empty"
    assert not (
        data_distribution == DataDistribution.DISCRETE and len(devices) > 1
    ), "multi GPU execution is only supported in pc_gaussian"

    graph = init_pc_graph(data)

    if data_distribution == DataDistribution.GAUSSIAN:
        (
            (skeleton, separation_sets, pmax, discover_skeleton_kernel_time),
            discover_skeleton_time,
        ) = discover_skeleton_gpu_gaussian(
            graph,
            data,
            gaussian_correlation_matrix,
            alpha,
            max_level,
            num_variables,
            num_observations,
            kernels=kernels,
            is_debug=is_debug,
            should_log=should_log,
            devices=devices,
            sync_device=sync_device,
        )
    elif data_distribution == DataDistribution.DISCRETE:
        (
            (skeleton, separation_sets, pmax, discover_skeleton_kernel_time),
            discover_skeleton_time,
        ) = discover_skeleton_gpu_discrete(
            graph,
            data,
            alpha,
            max_level,
            num_variables,
            num_observations,
            kernels=kernels,
            max_memory_size=discrete_max_memory_size,
            is_debug=is_debug,
            should_log=should_log,
        )

    (directed_graph, edge_orientation_time) = orient_edges(
        nx.DiGraph(skeleton), separation_sets
    )

    return PCResult(
        directed_graph,
        separation_sets,
        pmax,
        discover_skeleton_time,
        edge_orientation_time,
        discover_skeleton_kernel_time,
    )
