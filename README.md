![Python tests + coverage](https://github.com/hpi-epic/gpucsl/actions/workflows/test-python.yml/badge.svg)
![test benchmarks](https://github.com/hpi-epic/gpucsl/actions/workflows/test-benchmarks.yml/badge.svg)
![test cuda](https://github.com/hpi-epic/gpucsl/actions/workflows/test-cuda.yml/badge.svg)
![lint python](https://github.com/hpi-epic/gpucsl/actions/workflows/lint-python.yml/badge.svg)
![lint cuda](https://github.com/hpi-epic/gpucsl/actions/workflows/lint-cuda.yml/badge.svg)

# GPUCSL

`GPUCSL` is a python library for constraint-based causal structure learning using Graphics Processing Units (GPUs). In particular, it utilizes CUDA for GPU acceleration to speed up the well-known PC algorithm.

## Features

- Performant GPU implementation of the PC algorithm (covers discrete, and multivariate normal distributed data), see [General Notes](#notes);
- Multi-GPU support (experimental; for now, only for gaussian CI kernel);
- Easy to install and use, see [Usage](#usage);
- A Command Line Interface (CLI);
- Modular, extensible, tested thoroughly.

## <a name="notes"></a> General Notes

`GPUCSL` enables the GPU-accelerated estimation of the equivalence class of a data generating Directed Acyclic Graph (DAG) from observational data via constraint-based causal structure learning, cf. Kalisch et al. [^Kalisch] or Colombo and Maathuis [^Colombo]. Within the equivalence class, all DAGs have the same skeleton and the same v-structures and they can be uniquely represented by a Completely Partially Directed Acyclic Graph (CPDAG). In this context, `GPUCSL` implements the fully order-independent version of the PC algorithm, called PC-stable, to estimate the CPDAG under common faithfulness and sufficiency assumptions, see Colombo and Maathuis [^Colombo]. Hence, the implementation follows the `pc`-function within the R-package `pcalg` (For more information, see the [pcalg-Documentation](https://cran.r-project.org/web/packages/pcalg/pcalg.pdf#pc)). In particular, in the case of conflicts within the orientation phase, conflicts are solved similar to the `pc` within the `pcalg` implementation and yield matching results.

Note, that `GPUCSL` provides kernel implementations that cover conditional independence (CI) tests for discrete distributed data according to the ideas of Hagedorn and Huegle[^HagedornDiscrete]. Implementation of the CI tests for multivariate normal (or Gaussian) distributed data follows the ideas of Schmidt et al. [^SchmidtGaussian] and Zarebavani et al. [^ZarebavaniCupc].

## <a name="usage"></a>  Usage

Linux and a NVIDIA GPU with CUDA are required. We support running on multiple GPUs (experimental; for now, only for Gaussian CI kernel - `GaussianPC`).

### CLI

With the CLI, the PC algorithm is executed on the specified datasets. Three output files will be written to the specified directory:

- {dataset}.gml - the resulting CPDAG  containing the causal relationships
- {dataset}_pmax.csv - the maximum pvalues used for the conditional independence tests
- {dataset}_sepset.csv - the separation sets for the removed edges
- {dataset}_config.txt - the parameters the CLI got called with

All paths you give to the CLI are relative to your current directory.
An example call for `GPUCSL` with a CI test for multivariate normal or Gaussian distributed data could look like this (assuming your data is in "./data.csv"):

```bash
python3 -m gpucsl --gaussian -d ./data.csv -o . -l 3
```

### Python API

`GPUCSL` provides a python API for:

- `GaussianPC` - implements the full PC algorithm for multivariate normal data. Outputs the CPDAG from observational data. Similar to the CLI.
- `DiscretePC` -implements the full PC algorithm for discrete data. Outputs the CPDAG from observational data. Similar to the CLI.
- `discover_skeleton_gpu_gaussian` - determines the undirected skeleton graph for gaussian distribution
- `discover_skeleton_gpu_discrete` - determines the undirected skeleton graph for discrete distribution
- `orient_edges` - orients the edges of the undirected skeleton graph by detection of v-structures and application of Meek's orientation rules. Outputs the CPDAG from skeleton.

Additional detail is found in the [API description](https://github.com/hpi-epic/gpucsl/blob/main/docs/Public-api.md).

The following code snippet provides a small example for using `GaussianPC`:
```python
import numpy as np
from gpucsl.pc.pc import GaussianPC

samples = np.random.rand(1000, 10)
max_level = 3
alpha = 0.05
((directed_graph, separation_sets, pmax, discover_skeleton_runtime,
  edge_orientation_runtime, discover_skeleton_kernel_runtime),
    pc_runtime) = GaussianPC(samples,
                     max_level,
                     alpha).set_distribution_specific_options().execute()

```

Additional usage examples can be found in `docs/examples/`.

### Multi GPU support

Multi GPU support is currently only implemented for the gaussian CI kernel (`GaussianPC`) for skeleton discovery. The adjacency matrix (skeleton) is partitioned horizontally, and each GPU executes the CI tests on the assigned partition. For example, in the case of the dataset with 6 variables and 3 GPUs, the first GPU executes CI tests on edges 0-i, 1-i, where i is in {0..5\} (0-indexing), the second GPU executes CI tests on edges 2-i, 3-i and so on.

In case of an edge being deleted on multiple GPUs in the same level (for example, the edge 1-3 is deleted on the first GPU, the edge 3-1 is deleted on the second GPU in the example above), the separation set with the highest p-value is written to the end result (along with the corresponding p-value).

## Installation

- Install cuda toolkit (see also <https://docs.cupy.dev/en/stable/install.html>).
- Optional, but recommended: activate a virtual python environment, e.g. `python3 -m venv venv && . venv/bin/activate`
- Make sure to manually install the cupy version that matches your installed cuda version (<https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi>). For example, if cuda 11.2 is installed, run `pip install cupy-cuda112`.
- `pip install gpucsl`

## How to use with docker

First install Docker (for instructions refer to: <https://docs.docker.com/get-docker/>)
Please remember: you will still need a NVIDIA GPU and the CUDA Toolkit installed to used the Docker image.

The current dockerfiles are written with CUDA version 11.2 as a target. Should your host system have a different version installed
(you can look it up by running ```nvidia-smi -q -u``` and look for CUDA Version), you should go into the dockerfile and change the version
in line 1 (the from statement) and in line 29 where the correct cupy version is installed.

To just use gpucsl:

```
docker build --target gpucsl -t gpucsl .
docker run --runtime=nvidia --gpus all -i -t gpucsl
```

If you want to upload/download files from/to the container run (to get the container id look it up by: ```docker container ps```):

```
# upload your-file.txt to container from the current directory into the directory where gpucsl is installed
docker cp your-file.txt container_id:/gpucsl/your-file.txt

# download your-file.txt from container where gpucsl is installed to current directory
docker cp container_id:/gpucsl/your-file.txt your-file.txt
```

If you want to run the benchmarks:

```
docker build --target gpucsl-benchmarks -t benchmarks .
docker run --runtime=nvidia --gpus all -i -t benchmarks
```

## <a name="CLIdocumentation"></a> Full CLI Documentation

The following options are available:
| Shorthand  |  Longform |  Description |
|---|---|---|
| | --gaussian | Executes Fisher’s z-transformed (partial) correlation-based CI test for multvariate normal (or Gaussian) distributed data; Either this or discrete have to be set! |
| | --discrete | Executes Pearson’s χ2 CI test for discrete distributed data; Either this or gaussian have to be set! |
| -a | --alpha  | Sets the alpha level for PC algorithm to use. Default is 0.05. |
| -b | --debug   | Is a flag for debugging the kernels. When activated kernels get compiled with debug flag and lineinfo enabled. |
| -c | --correlation-matrix  | Defines the path to a cached correlation matrix file in csv format. You still have to give a dataset (-d/--dataset option)! |
| -d | --dataset   |  (required) Defines the path to your input data in csv format; Is relative to your current directory |
| -h | --help   |  shows a list and a sort explanation of all available options |
| -g | --gpus  | Indices of GPUs that should be used for independence test calculation (currently only supported for gaussian independence test). When not specified will use all available GPUs. |
| -l | --max-level  | Gives the max level for the PC algorithm (level of the pc algorithm is <= max level) |
| -o | --output-directory  |  (required) Defines the output directory; Is relative to your current directory |
| -s | --sync-device  | Index of the GPU used to sync the others (currently only supported for gaussian independence test). Default is 0 |
| -v | | Prints verbose output |
| -vv | | Prints debug output |

## Development

### Dependencies setup

- Clone the repo
- From the project dir, call `./scripts/download-data.sh` (see `download-data.sh` section in [scripts](#scripts)) to download the data folder from dropbox. It contains several example/test datasets.
- It is recommended to use a virtual environment: `python3 -m venv venv && source venv/bin/activate` (in bash, zsh).
- Upgrade pip and install build: `python3 -m pip install --upgrade pip build`.
- Make sure to install the cuda toolkit and cupy (see [installation](#installation)).
- Run `pip install .` to install the release package.
- Run `pip install -e "./[dev]"` to also install for development, with dev dependencies.

### Linting Setup

- Install clang-tidy and clang-format `sudo apt install clang-format`, `sudo apt install clang-tidy`.
- Set up git to use our hooks `git config --local core.hooksPath .githooks/` to execute lint checks on pushing.
- You can manually lint code with `./scripts/lint-python.sh` and `./scripts/lint-cuda.sh`.
- You can autofix code with `./scripts/lint-python.sh --fix` and `./scripts/lint-cuda.sh --fix`.
- Import `.vscode/settings.json` if you use vscode. This sets up VSCode to automatically run `black` and `clang-format` (respectively) on save.

### Building a Package

- Build sdist: `python3 -m build`.
- Build for dev: `python3 -m pip install --editable .` (installs it locally).

### Running Tests

- Make sure to have your python environment activated and run `python3 -m pip install -e "./[dev]"`.
- Run python tests: `pytest` (or in VSCode go to "Testing").
- There are additional flags:
  - for tests that take a long time and end-to-end tests from installation to CLI call, there is `pytest --run_slow`;
  - for CI only tests, there is `pytest --run_testPyPI`.
- Run cuda tests: `./scripts/test-cuda.sh`.

### Release a new version using the github action

- Bump the version in setup.cfg and commit your change to main.
- On this commit, create a new tag (e.g., `git tag v0.0.1` and `git push origin v0.0.1`)
- Go to <https://github.com/hpi-epic/gpucsl/releases> and create a new release for this tag with the same name as the version.
- A github action will run on release and first test this deployment on testPyPi and then publish the library to pypi.

### Manually publish to PyPi

 ```bash
 python3 -m pip install --upgrade build twine
 python3 -m build
 # Upload to testPyPi
 # use __token__ as username and the pypi token as password
 python3 -m twine upload --repository testpypi dist/*
 # Upload to PyPi
 python3 -m twine upload dist/*
 ```

## Scripts

We provide some helper scripts for data generation, testing, and benchmarking.

To run some of the scripts below, a working R installation is required.

### R setup

An R installation with the following packages is needed. These steps were tested with
R 4.1.2 on Ubuntu 20.04.3 LTS.

- install R: see <https://cran.r-project.org/bin/linux/ubuntu/>.

- install further required packages

```bash
apt install -y r-cran-fastica
apt install -y libv8-3.14-dev
apt install -y libcurl4-openssl-dev
apt install -y libgmp3-dev
```

- install R packages // TODO RScript to be tested

```R
install.packages("BiocManager")
BiocManager::install(version = "3.14")
BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
install.packages("pcalg")
install.packages("XML")
install.packages("tictoc")
install.packages("here")
install.packages("bnlearn")
install.packages("igraph")
install.packages("optparse")
```

### `use_pcalg_gaussian.R`

This can be used to run the R-package `pcalg` on test data and output its results (as comparison for gpucsl). Some of the outputs are already contained in the `data` folder.

Usage (make sure to do the call from the top-level project folder):

- `./scripts/use_pcalg_gaussian.R {dataset}`;
- for example: `./scripts/use_pcalg_gaussian.R NCI-60`.

Assumes that "NCI-60" folder lies in the `data` folder, also works for all other datasets in `data`.

### `download-data.sh`

Call this script once to download all test data from a dropbox folder (without having to generate some parts of it yourself which can take a long time). It will create a `data` folder. Please check dataset license information in [dropbox](https://www.dropbox.com/sh/t5jw5vbwg8gaoxt/AAA-oQ9FMp2a_Ou7JuhOMiVca?dl=0) `README.md` before downloading.

### `use_bnlearn_discrete.r`

This script can be used to generate discrete test data using the R-package bnlearn. The outputs necessary for the package tests can be generated by `./preprare-test-data.sh`, which internally calls this script.
The data needs to be in the data folder.
Usage:

- `Rscript use_bnlearn_discrete.r {dataset_name} {maximum_level}`;
- for example: `Rscript use_bnlearn_discrete.r alarm 3`.

Make sure to do the call from the scripts folder.

### `encode-discrete-data.py`

This is a simple wrapper around sklearn's OrdinalEncoder to allow csv data that is not already encoded to be loaded into the library.
Usage:

- `python3 -m encode-discrete-data.py {dataset_name}`;
- for example: `python3 -m encode-discrete-data.py alarm`.

### Benchmarks

- To execute, run `./benchmarks/run_benchmarks.sh`.
  - The script will tell you the output folder where it wrote the benchmark run results. It automatically installs R dependencies. You can also specify which benchmarks to run in the first parameter.
  - To run the benchmarks, you have to place the cupc repo in the benchmarks/cupc subdirectory. A forked version with timing annotations is available at <https://github.com/benrobby/cupc>.
- Once completed, open `benchmarks/visualization` notebooks (e.g. in VSCode just open the file, for instructions see <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>). In the notebook, specify the benchmark results folder (the one from step 1) and run all cells.

## Contributors

- Tom Braun (@BraunTom)
- Christopher Hagedorn (@ChristopherSchmidt89)
- Johannes Huegle (@JohannesHuegle)
- Ben Hurdelhey (@benrobby)
- Dominik Meier (@XPerianer)
- Petr Tsayun (@PeterTsayun)

## License

This project is licensed under the MIT license (see `LICENSE.txt`) unless otherwise noted in the respective source files:

- `gpucsl/pc/helpers.py`;

For the license information of datasets, please check `README.md` in [dropbox](https://www.dropbox.com/sh/t5jw5vbwg8gaoxt/AAA-oQ9FMp2a_Ou7JuhOMiVca?dl=0) before downloading datasets.

## References

[^Kalisch]: Kalisch M., Mächler M., Colombo D., Maathuis M.H., Bühlmann P. (2012). "Causal Inference Using Graphical Models with the R Package pcalg." Journal of Statistical Software, 47(11), pp. 1–26.
[^Colombo]: Colombo D., and Maathuis, M.H. (2014). "Order-independent constraint-based causal structure learning." Journal of Machine Learning Research 15 3921-3962.
[^HagedornDiscrete]: Hagedorn, C., and Huegle, J. (2021). "GPU-Accelerated Constraint-Based Causal Structure Learning for Discrete Data." Proceedings of the 2021 SIAM International Conference on Data Mining (SDM). pp. 37–45.
[^SchmidtGaussian]: Schmidt, C., Huegle, J., and Uflacker, M. (2018). "Order-independent constraint-based causal structure learning for gaussian distribution models using GPUs." Proceedings of the 30th International Conference on Scientific and Statistical Database Management (SSDBM). pp. 19:1–19:10.
[^ZarebavaniCupc]: Zarebavani, B., Jafarinejad, F., Hashemi, M., & Salehkaleybar, S. (2020). cuPC: CUDA-Based Parallel PC Algorithm for Causal Structure Learning on GPU. IEEE Transactions on Parallel and Distributed Systems, 31(3), 530–542.
