import logging
import cupy as cp

from gpucsl.cli.command_line_parser import build_command_line_parser
from gpucsl.cli.cli_io import (
    read_correlation_matrix,
    write_results_and_config,
    read_csv,
)
from gpucsl.cli.cli_util import warning, error
from gpucsl.pc.pc import DiscretePC, GaussianPC


def setLogLevel(level):
    if level == 1:
        logging.basicConfig(level=logging.INFO)
    if level > 1:
        logging.basicConfig(level=logging.DEBUG)


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

    is_gaussian = args.gaussian
    is_discrete = args.discrete

    max_gpu_count = cp.cuda.runtime.getDeviceCount() if is_gaussian else 1
    if gpus is None:
        gpus = [i for i in range(max_gpu_count)]

    if is_gaussian + is_discrete != 1:
        error("Please set exactly one of the options --gaussian/--discrete")

    if is_discrete and correlation_matrix_path is not None:
        error(
            "Discrete independence test does not use a correlation matrix. Please check the arguments you are giving to gpucsl!"
        )

    if is_discrete and len(gpus) > 1:
        error(
            "Multi GPU independece test currently only supported for gaussian independece test"
        )

    data = read_csv(dataset)

    if is_discrete:
        pc_class = DiscretePC
    elif is_gaussian:
        pc_class = GaussianPC

    pc = pc_class(data, max_level, alpha_level, is_debug=debug, should_log=should_log)

    if is_discrete:
        pc.set_distribution_specific_options()
    elif is_gaussian:
        correlation_matrix = read_correlation_matrix(correlation_matrix_path)

        pc.set_distribution_specific_options(gpus, sync_device, correlation_matrix)

    # result to ignore the timing informations that are only interesting for benchmarking
    result = pc.execute().result

    write_results_and_config(output_directory, dataset, command_line_arguments, result)
