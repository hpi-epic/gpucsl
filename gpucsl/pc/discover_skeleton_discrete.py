from timeit import default_timer as timer
from typing import List

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from gpucsl.pc.helpers import SkeletonResult

from gpucsl.pc.helpers import (
    postprocess_pmax,
    timed,
    log,
    log_time,
)
from gpucsl.pc.kernel_management import (
    Kernels,
    KernelGetter,
    bytes_to_giga_bytes,
)


@timed
def discover_skeleton_gpu_discrete(
    skeleton: np.ndarray,
    data: np.ndarray,
    alpha: float,
    max_level: int,
    num_variables: int,
    num_observations: int,
    kernels=None,
    memory_restriction: int = None,
    is_debug: bool = False,
    should_log: bool = False,
) -> SkeletonResult:

    enc = OrdinalEncoder(dtype=int)
    data = pd.DataFrame(enc.fit_transform(data))

    d_skeleton = cp.asarray(skeleton, dtype=cp.uint16)
    d_compacted_skeleton = d_skeleton.astype(np.int32, copy=True)

    # Load data in column-first style
    d_data = cp.asarray(data, order="F", dtype=cp.uint8)

    stream = cp.cuda.get_current_stream()

    d_data_dimensions = cp.asarray((cp.amax(data, axis=0) + 1), dtype=cp.uint8)

    d_seperation_sets = cp.full(num_variables * num_variables * max_level, -1, np.int32)

    d_pmax = cp.full((num_variables, num_variables), 0, np.float32)

    max_dim = cp.max(d_data_dimensions).get()

    start = timer()
    if kernels is None:
        kernels = Kernels.for_discrete_ci(
            max_level,
            num_variables,
            max_dim,
            num_observations,
            memory_restriction,
            is_debug,
            should_log,
        )

    log_time("Kernel compiling", timer() - start)

    max_memory_size = kernels.ci_test.max_memory_size

    # Allocate memory for contigency tables
    log(f"Allocate {bytes_to_giga_bytes(max_memory_size)}GB of memory")
    working_memory = cp.cuda.Memory(max_memory_size)

    log("Start kernels")
    computation_start = timer()

    for level in range(0, max_level + 1):
        log(f"LEVEL {level}")
        level_start = timer()

        if level != 0:
            kernels.compact(
                level,
                d_skeleton,
                d_compacted_skeleton,
                cp.uint32(0),
            )
            max_neigbours_count = d_compacted_skeleton[:, 0].max().get()

            if (max_neigbours_count - 1) < level:
                log("FINISH: maximal neighbours count is too small to proceed")
                break

        kernels.ci_test(
            level,
            d_data,
            d_skeleton,
            d_compacted_skeleton,
            d_data_dimensions,
            alpha,
            working_memory.ptr,
            d_seperation_sets,
            d_pmax,
        )
        stream.synchronize()
        log_time(
            f"Level {level} time",
            timer() - level_start,
        )

    computation_time = timer() - computation_start

    log("Getting data from GPU")
    gpu_transfer_start = timer()
    result_skeleton = d_skeleton.get()
    stream.synchronize()
    log_time("Data transport from GPU", timer() - gpu_transfer_start)

    separation_sets = d_seperation_sets.get().reshape(
        (num_variables, num_variables, max_level)
    )

    pmax = postprocess_pmax(d_pmax.get())

    return SkeletonResult(result_skeleton, separation_sets, pmax, computation_time)
