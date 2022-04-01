from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from gpucsl.pc.discover_skeleton_discrete import discover_skeleton_gpu_discrete
from gpucsl.pc.edge_orientation.edge_orientation import orient_edges
from gpucsl.pc.discover_skeleton_gaussian import discover_skeleton_gpu_gaussian

from gpucsl.pc.helpers import PCResult, init_pc_graph, timed
from gpucsl.pc.pc_validations import (
    validate_alpha,
    determine_max_level,
    validate_given_devices,
    warn_against_too_high_memory_consumption_gaussian_pc,
)


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
        sync_device: int = None,
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
        # Todo: change
        self.max_memory_size = max_memory_size
