from pathlib import Path
import pandas as pd
import argparse
import csv

from gpucsl.pc.pc import pc
from gpucsl.pc.kernel_management import (
    DiscreteKernel,
    Kernels,
    CompactKernel,
    GaussianKernel,
)
import cupy as cp
from timeit import default_timer as timer
from sklearn.preprocessing import OrdinalEncoder

from gpucsl.pc.helpers import correlation_matrix_of
from gpucsl.pc.pc import DataDistribution


SCRIPT_PATH = Path(__file__).parent.resolve()
DATASET_FOLDER_PATH = SCRIPT_PATH / ".." / "data"


def precompile_gaussian_and_compact_kernel_getters_for_devices(
    max_level, variable_count, devices
):
    n_devices = len(devices)

    kernels = [None] * n_devices
    device_streams = [None] * n_devices

    for device_index, device in enumerate(devices):
        with cp.cuda.Device(device):
            device_streams[device_index] = cp.cuda.Stream()
            with device_streams[device_index]:
                kernels[device_index] = Kernels.for_gaussian_ci(
                    variable_count, n_devices, max_level
                )

    for stream in device_streams:
        stream.synchronize()

    return kernels


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-dg",
        "--datasets_gaussian",
        required=False,
        default="",
        help="the dataset (gaussian) to run",
    )
    parser.add_argument(
        "-dd",
        "--datasets_discrete",
        required=False,
        default="",
        help="the dataset (discrete) to run",
    )
    parser.add_argument("-o", "--output_dir", help="output dir to store csv")
    parser.add_argument("-l", "--max_level", help="maximum level")
    parser.add_argument("--devices", help="device id list", default="0", type=str)
    parser.add_argument(
        "--sync_device",
        help="device, where P2P synchronization is done in multi-GPU case; ignored in a single GPU setting",
        type=int,
    )

    args = parser.parse_args()

    devices = [int(entry) for entry in args.devices.split(",")]
    sync_device = args.sync_device

    if sync_device is not None:
        assert sync_device in devices, "sync_device must be in devices list"

    def transform_dataset_names(names):
        names = [d.replace("'", "").replace('"', "") for d in names.split(",")]
        names = [n for n in names if n != ""]
        return names

    datasets_gaussian = transform_dataset_names(args.datasets_gaussian)
    datasets_discrete = transform_dataset_names(args.datasets_discrete)
    output_path = SCRIPT_PATH / args.output_dir
    max_level = int(args.max_level)
    return (
        datasets_gaussian,
        datasets_discrete,
        output_path,
        max_level,
        devices,
        sync_device,
    )


OUTPUT_CSV_HEADER = [
    "library",
    "dataset",
    "distribution",
    "full_runtime",
    "discover_skeleton_time",
    "edge_orientation_time",
    "kernel_time",
]


def run_benchmark_gaussian(samples, max_level, devices, sync_device):
    correlation_matrix = correlation_matrix_of(samples)
    start = timer()

    kernels = precompile_gaussian_and_compact_kernel_getters_for_devices(
        max_level, samples.shape[1], devices
    )

    (pc_result, pc_runtime) = pc(
        samples,
        DataDistribution.GAUSSIAN,
        max_level,
        0.05,
        gaussian_correlation_matrix=correlation_matrix,
        kernels=kernels,
        devices=devices,
        sync_device=sync_device,
    )

    duration_incl_compilation = timer() - start
    return (
        duration_incl_compilation,
        pc_runtime,
        pc_result,
    )


def run_benchmark_discrete(samples, max_level, devices, sync_device):
    start = timer()
    enc = OrdinalEncoder(dtype=int)
    data = pd.DataFrame(enc.fit_transform(samples))

    cp.asarray((cp.amax(data, axis=0) + 1), dtype=cp.uint8)

    kernels = Kernels.for_discrete_ci(
        max_level,
        samples.shape[1],
        cp.max(cp.asarray((cp.amax(data, axis=0) + 1), dtype=cp.uint8)).get(),
        samples.shape[0],
    )

    (pc_result, pc_runtime) = pc(
        samples, DataDistribution.DISCRETE, max_level, alpha=0.05, kernels=kernels
    )

    duration_incl_compilation = timer() - start
    return (
        duration_incl_compilation,
        pc_runtime,
        pc_result,
    )


def run_benchmarks_for_distribution(
    runtimes,
    runtimes_incl_compilation,
    dataset_names,
    max_level,
    run_benchmark_for_distrib,
    distribution_name: str,
    devices,
    sync_devics,
):
    for dataset in dataset_names:
        print(f"Benchmarking GPUCSL on dataset: {dataset}, max level {max_level}")

        dataset_path = DATASET_FOLDER_PATH / dataset
        samples_path = dataset_path / f"{dataset}.csv"
        samples = pd.read_csv(samples_path.absolute(), header=None).to_numpy()

        (duration_incl_compilation, pc_runtime, pc_result) = run_benchmark_for_distrib(
            samples, max_level, devices, sync_device
        )

        runtimes_incl_compilation.append(
            [
                "gpucsl_incl_compilation",
                dataset,
                distribution_name,
                duration_incl_compilation,
                pc_result.discover_skeleton_runtime,
                pc_result.edge_orientation_runtime,
                pc_result.discover_skeleton_kernel_runtime,
            ]
        )
        runtimes.append(
            [
                "gpucsl",
                dataset,
                distribution_name,
                pc_runtime,
                pc_result.discover_skeleton_runtime,
                pc_result.edge_orientation_runtime,
                pc_result.discover_skeleton_kernel_runtime,
            ]
        )

        print(
            f"GPUCSL on {dataset}: full_runtime={pc_runtime}, discover_skeleton_time={pc_result.discover_skeleton_runtime}, edge_orientation_time={pc_result.edge_orientation_runtime}"
        )


if __name__ == "__main__":
    (
        datasets_gaussian,
        datasets_discrete,
        output_path,
        max_level,
        devices,
        sync_device,
    ) = parse_arguments()

    print("GPUCSL gaussian Datasets: " + str(datasets_gaussian))
    print("GPUCSL discrete Datasets: " + str(datasets_discrete))

    runtimes = [list(OUTPUT_CSV_HEADER)]
    runtimes_incl_compilation = [list(OUTPUT_CSV_HEADER)]

    run_benchmarks_for_distribution(
        runtimes,
        runtimes_incl_compilation,
        datasets_gaussian,
        max_level,
        run_benchmark_gaussian,
        "gaussian",
        devices,
        sync_device,
    )

    run_benchmarks_for_distribution(
        runtimes,
        runtimes_incl_compilation,
        datasets_discrete,
        max_level,
        run_benchmark_discrete,
        "discrete",
        devices,
        sync_device,
    )

    with open(output_path / "gpucsl.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(runtimes)
    with open(output_path / "gpucsl_incl_compilation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(runtimes_incl_compilation)
