from gpucsl.cli.cli_util import error, warning
import cupy as cp


def validate_given_devices(devices, sync_device):
    max_gpu_count = cp.cuda.runtime.getDeviceCount()

    if len(devices) < 1 or len(devices) > max_gpu_count:
        error(
            f"GPU count should be between 1 and {max_gpu_count}. You specified: {len(devices)}"
        )

    for gpu_index in devices:
        if gpu_index < 0 or gpu_index >= max_gpu_count:
            error(
                f"Specified gpu indices should be between 0 and {max_gpu_count - 1}. You specified: {gpu_index}"
            )

    if sync_device is not None and sync_device not in devices:
        error(
            f"The sync device has to be one of the specified gpus. You gave gpus: {', '.join(devices)} and sync device: {sync_device}"
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
    max_possible_level = variable_count - 2

    if max_level is None:
        max_level = max_possible_level

        if max_level < 0:
            error(f"Max level should be >= 0. Your input: {max_level}")

    if max_level > max_possible_level:
        warning(
            f"You set the max level to {max_level}, but the biggest possible level is {max_possible_level}. The pc algorithm will at a maximum only run until level {max_possible_level}."
        )
        max_level = max_possible_level  # we never run more than possible_max_level and we want to keep it small so we do not have to allocate too much unnecessary memory

    return max_level
