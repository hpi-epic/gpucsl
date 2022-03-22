import logging
import cupy as cp

from gpucsl.cli.command_line_parser import build_command_line_parser
from gpucsl.cli.cli_io import (
    read_correlation_matrix,
    write_results_and_config,
    read_csv,
)
from gpucsl.cli.cli_util import warning, error
from gpucsl.pc.pc import DataDistribution, pc


def setLogLevel(level):
    if level == 1:
        logging.basicConfig(level=logging.INFO)
    if level > 1:
        logging.basicConfig(level=logging.DEBUG)


def gaussian_pc_memory_warning(max_level: int):
    gpu_data = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())

    total_threads = (
        gpu_data["multiProcessorCount"] * gpu_data["maxThreadsPerMultiProcessor"]
    )
    # total threads * estimate of data entries allocated in kernels * sizeof datatype of data = int = 4
    estimated_memory_consumption = total_threads * (5 * max_level**2) * 4

    if estimated_memory_consumption > gpu_data["totalGlobalMem"]:
        warning(
            "if the algorithm does not abort it could be that it tries to allocate to much memory on higher levels and errors. Please consider to set the maximum level manually via the -l/--level flag. "
        )


# for remaining_pc_arguments and pc_keyword_arguments just provide the normal parameters
# you would provide to your choosen pc_variant, without data and max_level
def run_on_dataset(
    data_distribution: DataDistribution,
    max_level_warning,
    dataset_path,
    max_level,
    *remaining_pc_arguments,
    **pc_keyword_arguments,
):
    data = read_csv(dataset_path)
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

        max_level_warning(max_level)

    return pc(
        data,
        data_distribution,
        max_level,
        *remaining_pc_arguments,
        **pc_keyword_arguments,
    ).result


def gpucsl_cli(command_line_arguments):
    parser = build_command_line_parser()

    args = parser.parse_args(command_line_arguments)
    dataset = args.dataset

    setLogLevel(args.verbose)
    should_log = args.verbose > 0

    debug = args.debug

    max_level = args.max_level
    alpha_level = args.alpha
    output_directory = args.output_directory
    correlation_matrix_path = args.correlation_matrix

    sync_device = args.sync_device
    gpus = args.gpus
    max_gpu_count = cp.cuda.runtime.getDeviceCount()

    is_gaussian = args.gaussian
    is_discrete = args.discrete

    if is_gaussian + is_discrete != 1:
        error("Please set exactly one of the options --gaussian/--discrete")

    if is_discrete and correlation_matrix_path is not None:
        error(
            "Discrete independence test does not use a correlation matrix. Please check the arguments you are giving to gpucsl!"
        )

    if gpus is None:
        gpus = [i for i in range(max_gpu_count)]

    if is_discrete and len(gpus) > 1:
        error(
            "Multi GPU independece test currently only supported for gaussian independece test"
        )

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

    if alpha_level < 0 or alpha_level > 1:
        error("Alpha level has to be between 0 and 1")

    if is_discrete:
        data_distribution = DataDistribution.DISCRETE
        correlation_matrix = None
        memory_warning = lambda _: None
    elif is_gaussian:
        data_distribution = DataDistribution.GAUSSIAN
        correlation_matrix = read_correlation_matrix(correlation_matrix_path)
        memory_warning = gaussian_pc_memory_warning

    result = run_on_dataset(
        data_distribution,
        memory_warning,
        dataset,
        max_level,
        alpha_level,
        gaussian_correlation_matrix=correlation_matrix,
        is_debug=debug,
        should_log=should_log,
        devices=gpus,
        sync_device=sync_device,
    )

    write_results_and_config(output_directory, dataset, command_line_arguments, result)
