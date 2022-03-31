from enum import Enum, auto
from typing import List, NamedTuple
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from gpucsl.pc.edge_orientation.edge_orientation import orient_edges
from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian

from gpucsl.pc.helpers import PCResult, init_pc_graph, timed
from gpucsl.cli.cli_util import error, warning
import cupy as cp


def validate_given_devices(gpus, sync_device):
    max_gpu_count = cp.cuda.runtime.getDeviceCount()

    print("----------------")
    print(max_gpu_count, gpus, sync_device)

    if len(gpus) < 1 or len(gpus) > max_gpu_count:
        error(
            f"GPU count should be between 1 and {max_gpu_count}. You specified: {len(gpus)}"
        )

    for gpu_index in gpus:
        if gpu_index < 0 or gpu_index >= max_gpu_count:
            error(
                f"Specified gpu indices should be between 0 and {max_gpu_count - 1}. You specified: {gpu_index}"
            )

    if sync_device not in gpus:
        error(
            f"The sync device has to be one of the specified gpus. You gave gpus: {', '.join(gpus)} and sync device: {sync_device}"
        )


def validate_alpha(alpha):
    if alpha < 0 or alpha > 1:
        error("Alpha level has to be between 0 and 1")


def warn_against_too_high_memory_consumption_gaussian_pc(max_level: int):
    gpu_data = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())

    total_threads = (
        gpu_data["multiProcessorCount"] * gpu_data["maxThreadsPerMultiProcessor"]
    )
    # total threads * estimate of data entries allocated in kernels * sizeof datatype of data = int = 4
    estimated_memory_consumption = total_threads * (5 * max_level**2) * 4

    if estimated_memory_consumption > gpu_data["totalGlobalMem"]:
        warning(
            "if the algorithm does not terminate, a possible reason could be running out of memory on higher levels, and errors. Please consider setting the maximum level manually via the -l/--level flag. "
        )


def determine_max_level(max_level, data):
    variable_count = data.shape[1]

    possible_max_level = variable_count - 2
    if max_level is not None and max_level > possible_max_level:
        warning(
            f"You set the max level to {max_level}, but the biggest possible level is {possible_max_level}. The pc algorithm will at a maximum only run until level {possible_max_level}."
        )
        max_level = possible_max_level  # we never run more than possible_max_level and we want to keep it small so we do not have to allocate too much unnecessary memory

    if max_level is None:
        max_level = possible_max_level

        if max_level < 0:
            error(f"Max level should be >= 0. Your input: {max_level}")

    return max_level


class PC(ABC):
    def __init__(
        self,
        data: np.ndarray,
        max_level: int,
        alpha=0.05,
        kernels=None,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        validate_alpha(alpha)

        self.data = data
        self.max_level = determine_max_level(max_level, data)
        self.alpha = alpha
        self.kernels = kernels
        self.is_debug = is_debug
        self.should_log = should_log

        self.num_variables = data.shape[1]
        self.num_observations = data.shape[0]
        self.graph = init_pc_graph(data)

    @abstractmethod
    def skeleton_discovery_function(self):
        pass

    def discover_skeleton(self):
        return self.skeleton_discovery_function()(self)

    @abstractmethod
    def set_distribution_specific_options(self):
        pass

    @timed
    def execute(self):
        (
            (skeleton, separation_sets, pmax, discover_skeleton_kernel_time),
            discover_skeleton_time,
        ) = self.discover_skeleton()

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


# simple glue method so we do not have to change discover_skeleton_gpu_gaussian
def discover_skeleton_gpu_gaussian_using_config(config):
    return discover_skeleton_gpu_gaussian(
        config.graph,
        config.data,
        config.correlation_matrix,
        config.alpha,
        config.max_level,
        config.num_variables,
        config.num_observations,
        config.kernels,
        config.is_debug,
        config.should_log,
        config.devices,
        config.sync_device,
    )


class GaussianPC(PC):
    def skeleton_discovery_function(self):
        return discover_skeleton_gpu_gaussian_using_config

    def set_distribution_specific_options(
        self,
        devices: List[int] = [0],
        sync_device: int = 0,
        correlation_matrix: np.ndarray = None,
    ):
        warn_against_too_high_memory_consumption_gaussian_pc(self.max_level)
        assert len(devices) > 0, "Invalid device list, must not be empty"
        validate_given_devices(devices, sync_device)

        self.devices = devices
        self.sync_device = sync_device
        self.correlation_matrix = correlation_matrix


# simple glue method so we do not have to change discover_skeleton_gpu_discrete
def discover_skeleton_gpu_discrete_using_config(config):
    return discover_skeleton_gpu_discrete(
        config.graph,
        config.data,
        config.alpha,
        config.max_level,
        config.num_variables,
        config.num_observations,
        config.max_memory_size,
        config.kernels,
        config.memory_restriction,
        config.is_debug,
        config.should_log,
    )


class DiscretePC(PC):
    def skeleton_discovery_function(self):
        return discover_skeleton_gpu_discrete_using_config

    def set_distribution_specific_options(self, max_memory_size=None):
        self.max_memory_size = max_memory_size
