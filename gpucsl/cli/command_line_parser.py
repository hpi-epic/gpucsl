import argparse


def build_command_line_parser():
    parser = argparse.ArgumentParser(
        description="Library for causal structure learning on GPUs."
    )
    parser.add_argument(
        "-a", "--alpha", help="Sets the alpha level", default=0.05, type=float
    )
    parser.add_argument(
        "-b",
        "--debug",
        help="Compiles the cuda kernels with debug flag enabled",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--correlation-matrix", help="Path to a cached correlation matrix"
    )
    parser.add_argument(
        "-d", "--dataset", help="The dataset to run", default="", required=True
    )
    parser.add_argument(
        "--discrete",
        help="Uses a discrete conditional independence test",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--gaussian",
        help="Uses a gaussian conditional independence test for data with a multivariate distribution",
        action="count",
        default=0,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        help="Indices of GPUs that should be used for independence test calculation (currently only supported for gaussian independence test). When not specified will use all available GPUs.",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "-l",
        "--max-level",
        help="Maximum size of seperation set used in pc algorithm",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="Output dir to store output files",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sync-device",
        help="Index of the GPU used to sync the others (currently only supported for gaussian independence test). Defaults to 0",
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Prints debug messages to stdout. You can either use -v or -vv for an higher logging level (if you append more v this is counted as -vv). -v -> logs info; -vv -> logs debug info",
        action="count",
        default=0,
    )

    return parser
