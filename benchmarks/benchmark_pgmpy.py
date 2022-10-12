from pgmpy.estimators import PC
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from timeit import default_timer as timer

import argparse
import csv


SCRIPT_PATH = Path(__file__).parent.resolve()
DATASET_FOLDER_PATH = SCRIPT_PATH / ".." / "data"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dd",
        "--datasets_discrete",
        required=False,
        default="",
        help="the dataset (discrete) to run",
    )
    parser.add_argument("-o", "--output_dir", help="output dir to store csv")
    parser.add_argument("-l", "--max_level", help="maximum level")
    parser.add_argument("-n", "--n_jobs", help="Number of concurrent jobs")

    args = parser.parse_args()

    def transform_dataset_names(names):
        names = [d.replace("'", "").replace('"', "") for d in names.split(",")]
        names = [n for n in names if n != ""]
        return names

    datasets_discrete = transform_dataset_names(args.datasets_discrete)
    output_path = SCRIPT_PATH / args.output_dir
    max_level = int(args.max_level)

    n_jobs = int(args.n_jobs)

    return (
        datasets_discrete,
        output_path,
        max_level,
        n_jobs,
    )



def run_benchmark_discrete(samples, max_level, n_jobs = 40):
    enc = OrdinalEncoder(dtype=int)
    data = pd.DataFrame(enc.fit_transform(samples))

    model = PC(data)

    start = timer()
    skeleton, separation_sets = model.build_skeleton(significance_level=0.05, max_cond_vars=max_level, variant="parallel", n_jobs=n_jobs)

    discover_skeleton_end = timer()
    discover_skeleton_duration = discover_skeleton_end - start

    pdag = model.skeleton_to_pdag(skeleton, separation_sets)
    
    orient_edges_end = timer()
    orient_edges_duration = orient_edges_end - discover_skeleton_end

    return (
        pdag,
        discover_skeleton_duration,
        orient_edges_duration,
    )

def run_benchmarks_for_distribution(
    runtimes,
    dataset_names,
    max_level,
    n_jobs,
):
    for dataset in dataset_names:
        print(f"Benchmarking pgmpy on dataset: {dataset}, max level {max_level}")

        dataset_path = DATASET_FOLDER_PATH / dataset
        samples_path = dataset_path / f"{dataset}.csv"
        samples = pd.read_csv(samples_path.absolute(), header=None).to_numpy()

        (pdag, discover_skeleton_duration, orient_edges_duration) = run_benchmark_discrete(
            samples, max_level, n_jobs=n_jobs,
        )

        runtimes.append(
            [
                "pgmpy",
                dataset,
                "discrete",
                discover_skeleton_duration + orient_edges_duration,
                discover_skeleton_duration,
                orient_edges_duration,
                0,
            ]
        )

        print(
            f"pgmpy on {dataset}: full_runtime={discover_skeleton_duration + orient_edges_duration}, discover_skeleton_time={discover_skeleton_duration}, edge_orientation_time={orient_edges_duration}"
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

if __name__ == "__main__":
    (
        datasets_discrete,
        output_path,
        max_level,
        n_jobs,
    ) = parse_arguments()

    print("pgmpy discrete Datasets: " + str(datasets_discrete))

    runtimes = [list(OUTPUT_CSV_HEADER)]

    run_benchmarks_for_distribution(
        runtimes,
        datasets_discrete,
        max_level,
        n_jobs,
    )


    with open(output_path / "pgmpy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(runtimes)