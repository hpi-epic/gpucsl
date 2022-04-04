from timeit import default_timer as timer
from typing import List
from math import ceil

import cupy as cp
import numpy as np

from gpucsl.pc.kernel_management import Kernels

from gpucsl.pc.gaussian_device_manager import (
    GaussianDeviceManager,
    create_gaussian_device_manager,
)
from gpucsl.pc.helpers import (
    SkeletonResult,
    get_gaussian_thresholds,
    get_max_neighbor_count,
)

from gpucsl.pc.helpers import (
    correlation_matrix_of,
    timed,
    log,
    log_time,
)

from gpucsl.pc.kernel_management import KernelGetter


def gaussian_ci_worker(device_index: int, device_manager: GaussianDeviceManager):
    (
        variable_count,
        max_level,
        thresholds,
        num_observations,
        stop_flags,
        n_devices,
        devices,
    ) = device_manager.get_static_data()

    (
        d_skeleton,
        d_compacted_skeleton,
        d_correlation_matrix,
        d_zmin,
        d_seperation_sets,
        stream,
        ready_event,
        main_event,
        kernels,
    ) = device_manager.get_data_for_device_index(device_index)

    device = devices[device_index]
    columns_per_device = int(ceil(variable_count / len(devices)))

    log(f"[T: {device_index} D: {device}] Start")

    with cp.cuda.Device(device):
        with stream:
            for level in range(0, max_level + 1):
                level_start = timer()
                log(f"[T: {device_index} D: {device}] Level {level} start")

                d_skeleton = device_manager.get_skeleton_for_device_index(
                    device_index
                )  # get merged skeleton when using multiple devices

                if level != 0:
                    kernels.compact(
                        level,
                        d_skeleton,
                        d_compacted_skeleton,
                        cp.uint32(device_index),
                    )

                    max_neigbours_count = get_max_neighbor_count(
                        d_compacted_skeleton, columns_per_device, device_index
                    )

                    if (max_neigbours_count - 1) < level:
                        log(
                            f"[T: {device_index} D: {device}] FINISH: maximal neighbors count is too small to proceed"
                        )
                        stop_flags[device_index] = True
                        break

                kernels.ci_test(
                    level,
                    d_correlation_matrix,
                    d_skeleton,
                    d_compacted_skeleton,
                    cp.float_(thresholds[level]),
                    d_seperation_sets,
                    d_zmin,
                    cp.int32(device_index),
                )

                stream.synchronize()
                log_time(
                    f"[T: {device_index} D: {device}]    Level {level} time",
                    timer() - level_start,
                )

                # Synchronization when using multiple GPUs
                if n_devices > 1:
                    ready_event.set()
                    main_event.wait()
                    main_event.clear()

            stream.synchronize()
            ready_event.set()

    log(f"[T: {device_index} D: {device}] DONE")


@timed
def discover_skeleton_gpu_gaussian(
    skeleton: np.ndarray,
    data: np.ndarray,
    correlation_matrix: np.ndarray,
    alpha: float,
    max_level: int,
    num_variables: int,
    num_observations: int,
    kernels: Kernels = None,
    is_debug: bool = False,
    should_log: bool = False,
    devices: List[int] = [0],
    sync_device: int = None,
) -> SkeletonResult:

    thresholds = get_gaussian_thresholds(data, alpha)

    if correlation_matrix is None:
        correlation_matrix = correlation_matrix_of(data)

    device_manager, initialization_time = create_gaussian_device_manager(
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

    log_time("Initialization time", initialization_time)

    final_skeleton, computation_time = device_manager.compute_skeleton(
        gaussian_ci_worker
    )
    log_time("Computation time", computation_time)

    (
        (
            postprocessed_merged_pmax,
            merged_separation_sets,
        ),
        merge_time,
    ) = device_manager.get_merged_pmaxes_and_separation_sets()
    log_time("Merge time", merge_time)

    separation_sets = merged_separation_sets.reshape(
        (num_variables, num_variables, max_level)
    )

    return SkeletonResult(
        final_skeleton, separation_sets, postprocessed_merged_pmax, computation_time
    )
