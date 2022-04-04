import cupy as cp
import numpy as np
import sys
from threading import Event, Thread
from typing import List, Tuple, Callable

from gpucsl.pc.helpers import transform_to_pmax_cupy, postprocess_pmax_cupy, timed
from gpucsl.pc.kernel_management import Kernels

WorkerFunction = Callable[
    [int, "GaussianDeviceManager"], None
]  # GaussianDeviceManager is not defined at that point of time

# Wrapper to measure the initialization time of GaussianDeviceManager
@timed
def create_gaussian_device_manager(
    skeleton: np.ndarray,
    correlation_matrix: np.ndarray,
    thresholds: List[float],
    num_observations: int,
    max_level: int,
    devices: List[int],
    sync_device: int = None,
    kernels: Kernels = None,
    is_debug: bool = False,
    should_log: bool = False,
):
    return GaussianDeviceManager(
        skeleton,
        correlation_matrix,
        thresholds,
        num_observations,
        max_level,
        devices,
        sync_device,
        kernels,
        is_debug,
        should_log,
    )


# GaussianDeviceManager:
#       - allocates ressources on the corresponding devices
#       - compiles kernels on the corresponding devices
#       - performs synchronization in the multi-GPU setting

# sync_device: device, where P2P synchronization is done in multi-GPU case; ignored in a single GPU setting


class GaussianDeviceManager:
    def __init__(
        self,
        skeleton: np.ndarray,
        correlation_matrix: np.ndarray,
        thresholds: List[float],
        num_observations: int,
        max_level: int,
        devices: List[int],
        sync_device: int = None,
        kernels: Kernels = None,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        self.sync_device = sync_device
        self.devices = devices
        self.n_devices = len(devices)
        self.max_level = max_level
        self.variable_count = skeleton.shape[0]
        self.num_observations = num_observations
        self.thresholds = thresholds
        self.is_debug = is_debug
        self.should_log = should_log
        self.multi_gpu_execution = self.n_devices > 1

        self._initialize_sync_device_index_ressources()
        self._initialize_device_ressources()
        self._fill_device_ressources(skeleton, correlation_matrix)

        self.kernels = kernels
        self._initialize_kernels()

    def _initialize_sync_device_index_ressources(self):
        self.sync_device_index = (
            self.devices.index(self.sync_device) if self.sync_device is not None else 0
        )
        assert (self.sync_device_index >= 0) and (
            self.sync_device_index < self.n_devices
        ), "Invalid sync_device_index value"

        self.remaining_device_indexes = list(range(0, self.n_devices))
        self.remaining_device_indexes.remove(self.sync_device_index)

    def _initialize_device_ressources(self):
        self.d_skeletons = [None for _ in range(self.n_devices)]
        self.d_compacted_skeletons = [None for _ in range(self.n_devices)]
        self.d_correlation_matrices = [None for _ in range(self.n_devices)]

        self.d_seperation_sets_array = [None for _ in range(self.n_devices)]
        self.d_zmins = [None for _ in range(self.n_devices)]

        self.device_streams = [None for _ in range(self.n_devices)]
        self.stop_flags = [False for _ in range(self.n_devices)]
        self.ready_events = [Event() for _ in range(self.n_devices)]
        self.main_events = [Event() for _ in range(self.n_devices)]

    def _fill_device_ressources(
        self, skeleton: np.ndarray, correlation_matrix: np.ndarray
    ):
        for device_index, device in enumerate(self.devices):
            with cp.cuda.Device(device):
                self.device_streams[device_index] = cp.cuda.Stream()
                with self.device_streams[device_index]:

                    self.d_correlation_matrices[device_index] = cp.asarray(
                        correlation_matrix
                    )
                    d_skeleton = cp.asarray(skeleton)
                    self.d_skeletons[device_index] = d_skeleton
                    self.d_compacted_skeletons[device_index] = d_skeleton.astype(
                        np.int32, copy=True
                    )
                    self.d_seperation_sets_array[device_index] = cp.full(
                        self.variable_count * self.variable_count * self.max_level,
                        -1,
                        np.int32,
                    )
                    self.d_zmins[device_index] = cp.full(
                        (self.variable_count, self.variable_count),
                        sys.float_info.max,
                        np.float32,
                    )

        self.sync_streams()

    def _initialize_kernels(self):
        if self.kernels is not None:
            assert len(self.kernels) == self.n_devices
        else:
            self.kernels = [None] * self.n_devices
            self._compile_kernels(compile_kernels)

    def _compile_kernels(self, compile_function: Callable):
        threads = [
            Thread(target=compile_function, args=(device_index, self))
            for device_index in range(self.n_devices)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def sync_streams(self):
        for stream in self.device_streams:
            stream.synchronize()

    def get_static_data(self):
        return (
            self.variable_count,
            self.max_level,
            self.thresholds,
            self.num_observations,
            self.stop_flags,
            self.n_devices,
            self.devices,
        )

    def get_data_for_device_index(self, device_index: int):
        return (
            self.d_skeletons[device_index],
            self.d_compacted_skeletons[device_index],
            self.d_correlation_matrices[device_index],
            self.d_zmins[device_index],
            self.d_seperation_sets_array[device_index],
            self.device_streams[device_index],
            self.ready_events[device_index],
            self.main_events[device_index],
            self.kernels[device_index],
        )

    def get_skeleton_for_device_index(self, device_index: int):
        return self.d_skeletons[device_index]

    @timed
    def compute_skeleton(self, worker_function: WorkerFunction) -> np.ndarray:
        if self.multi_gpu_execution:
            final_skeleton = self.execute_ci_workers_in_parallel(worker_function)
        else:
            worker_function(0, self)
            final_skeleton = self.get_skeleton_for_device_index(0).get()

        return final_skeleton

    def execute_ci_workers_in_parallel(
        self, worker_function: WorkerFunction
    ) -> np.ndarray:
        threads = [
            Thread(target=worker_function, args=(device_index, self))
            for device_index in range(self.n_devices)
        ]

        for thread in threads:
            thread.start()

        for level in range(0, self.max_level + 1):
            if sum(self.stop_flags) == self.n_devices:
                break

            for device_index, ready_event in enumerate(self.ready_events):
                if not self.stop_flags[device_index]:
                    ready_event.wait()

            for ready_event in self.ready_events:
                ready_event.clear()

            d_merged_skeleton = self.synchronize_skeletons()

            for main_event in self.main_events:
                main_event.set()

        for thread in threads:
            thread.join()

        final_skeleton = d_merged_skeleton.get()

        return final_skeleton

    def synchronize_skeletons(self) -> cp.ndarray:
        d_merged_skeleton = self.merge_skeletons()
        self.broadcast_merged_skeleton(d_merged_skeleton)

        return d_merged_skeleton

    def merge_skeletons(self) -> cp.ndarray:
        d_merged_skeleton = self.d_skeletons[self.sync_device_index]

        if self.multi_gpu_execution:
            with cp.cuda.Device(self.devices[self.sync_device_index]):
                with self.device_streams[self.sync_device_index]:
                    for device_index in self.remaining_device_indexes:
                        d_merged_skeleton = cp.minimum(
                            # NOTE: arrays reside on different devices; cupy automatically enables P2P access
                            d_merged_skeleton,
                            self.d_skeletons[device_index],
                        )

        self.sync_streams()

        return d_merged_skeleton

    def broadcast_merged_skeleton(self, d_merged_skeleton: cp.ndarray):
        assert (
            self.d_skeletons[self.sync_device_index].device == d_merged_skeleton.device
        )
        # d_merged_skeleton already resides on the sync_device
        self.d_skeletons[self.sync_device_index] = d_merged_skeleton

        for device_index in self.remaining_device_indexes:
            with cp.cuda.Device(self.devices[device_index]):
                with self.device_streams[device_index]:
                    # move a copy of d_merged_skeleton to the corresponding device
                    self.d_skeletons[device_index] = d_merged_skeleton.copy()

    def merge_zmins(self) -> cp.ndarray:
        d_merged_zmin = self.d_zmins[self.sync_device_index]

        if self.multi_gpu_execution:
            with cp.cuda.Device(self.devices[self.sync_device_index]):
                with self.device_streams[self.sync_device_index]:
                    for device_index in self.remaining_device_indexes:
                        d_merged_zmin = cp.minimum(
                            # NOTE: arrays reside on different devices; cupy automatically enables P2P access
                            d_merged_zmin,
                            self.d_zmins[device_index],
                        )

            self.sync_streams()

        return d_merged_zmin

    # In d_zmin, only the upper right triangle is filled. This operation fills the whole
    # matrix using mirroring along the diagonal.
    def mirror_array(self, d_zmin: cp.ndarray) -> cp.ndarray:
        return cp.minimum(d_zmin, cp.transpose(d_zmin))

    def create_merge_masks(self, d_merged_zmin: cp.ndarray) -> List[cp.ndarray]:
        d_merge_masks = [None for _ in range(self.n_devices)]

        with cp.cuda.Device(self.devices[self.sync_device_index]):
            with self.device_streams[self.sync_device_index]:
                d_mirrored_merged_zmin = self.mirror_array(d_merged_zmin)

        for device_index, device in enumerate(self.devices):
            with cp.cuda.Device(self.devices[device_index]):
                with self.device_streams[device_index]:
                    # NOTE: arrays reside on different devices; cupy automatically enables P2P access
                    d_current_mirrored_zmin = self.mirror_array(
                        self.d_zmins[device_index]
                    )
                    d_merge_masks[device_index] = (
                        d_mirrored_merged_zmin == d_current_mirrored_zmin
                    )

        return d_merge_masks

    def merge_separation_sets(self, d_merge_masks: cp.ndarray) -> np.ndarray:
        # When the same edge was deleted on different devices, return the separation set with the highest pmax value
        d_merged_seperation_sets = self.d_seperation_sets_array[self.sync_device_index]

        with cp.cuda.Device(self.devices[self.sync_device_index]):
            with self.device_streams[self.sync_device_index]:

                for device_index in self.remaining_device_indexes:
                    d_current_separation_sets = self.d_seperation_sets_array[
                        device_index
                    ]
                    d_current_merge_mask = d_merge_masks[device_index]

                    repeated_mask = cp.repeat(d_current_merge_mask, self.max_level)

                    d_merged_seperation_sets[repeated_mask] = d_current_separation_sets[
                        repeated_mask
                    ]

        self.sync_streams()

        return d_merged_seperation_sets.get()

    def transform_merged_zmin_to_postprocessed_pmax(
        self, d_merged_zmin: cp.ndarray
    ) -> np.ndarray:
        with cp.cuda.Device(self.devices[self.sync_device_index]):
            with self.device_streams[self.sync_device_index]:
                pmax = postprocess_pmax_cupy(
                    transform_to_pmax_cupy(d_merged_zmin, self.num_observations)
                )

        return pmax.get()

    @timed
    def get_merged_pmaxes_and_separation_sets(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.multi_gpu_execution:
            d_merged_zmin = self.merge_zmins()
            d_merge_masks = self.create_merge_masks(d_merged_zmin)

            merged_separation_sets = self.merge_separation_sets(d_merge_masks)
            postprocessed_merged_pmax = (
                self.transform_merged_zmin_to_postprocessed_pmax(d_merged_zmin)
            )
        else:
            postprocessed_merged_pmax = (
                self.transform_merged_zmin_to_postprocessed_pmax(self.d_zmins[0])
            )
            merged_separation_sets = self.d_seperation_sets_array[0].get()

        return postprocessed_merged_pmax, merged_separation_sets


# Precompile kernels to separate the kernel compilation and kernel execution stages
# which allows separate measurements
def compile_kernels(device_index: int, device_manager: GaussianDeviceManager):
    with cp.cuda.Device(device_manager.devices[device_index]):
        with device_manager.device_streams[device_index]:
            device_manager.kernels[device_index] = Kernels.for_gaussian_ci(
                device_manager.variable_count,
                device_manager.n_devices,
                device_manager.max_level,
                device_manager.should_log,
                device_manager.is_debug,
            )

        device_manager.device_streams[device_index].synchronize()
