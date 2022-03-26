import os
from pathlib import Path

import pytest


test_folder = Path("./test_gpucsl_installation")
test_dataset_folder = Path(f"./data/")
output_folder = "output"


def execute_and_check_status(command: str):
    status = os.system(command)

    # command exist without error
    assert status == 0


def assert_file_exists(path):
    assert os.path.isfile(path)


def assert_folder_exists(path):
    assert os.path.isdir(path)


def gpucsl_cli_command(ci_test, dataset_name):
    dataset_path = (
        test_dataset_folder / dataset_name / (dataset_name + ".csv")
    ).absolute()
    return (
        f"python3 -m gpucsl {ci_test} "
        + f"-d {dataset_path} "
        + f"-o {(test_folder / output_folder).absolute()}"
    )


def assert_gpucsl_execution(
    ci_test: str, dataset_name: str, to_be_generated_files_suffixes
):
    execute_and_check_status(
        f"""
        python3 -m pip install cupy-cuda112 &&
        python3 -m pip install . &&
        mkdir -p {test_folder} && cd {test_folder} && rm -rf * &&
        {gpucsl_cli_command(ci_test, dataset_name)}
        """
    )

    assert_folder_exists(test_folder / output_folder)
    for file_suffix in to_be_generated_files_suffixes:
        assert_file_exists(test_folder / output_folder / (dataset_name + file_suffix))


@pytest.mark.run_slow
def test_gaussian_gpucsl_local_installation():
    assert_gpucsl_execution(
        "--gaussian",
        "coolingData",
        [".gml", "_sepset.txt", "_pmax.csv", "_config.txt"],
    )


@pytest.mark.run_slow
def test_gpucsl_run_with_level_set_to_0():
    assert_gpucsl_execution(
        "--gaussian -l 0",
        "coolingData",
        [".gml", "_sepset.txt", "_pmax.csv", "_config.txt"],
    )


@pytest.mark.run_slow
def test_discrete_gpucsl_local_installation():
    assert_gpucsl_execution(
        "--discrete -l 3",
        "alarm",
        [".gml", "_sepset.txt", "_pmax.csv", "_config.txt"],
    )


@pytest.mark.run_testPyPI
def test_gpucsl_testPyPI():
    execute_and_check_status(
        f"""
        python3 -m pip install . &&
        python3 -m pip install cupy-cuda112 &&
        mkdir -p {test_folder} && cd {test_folder} && rm -rf * &&
        python3 -m pip uninstall -y gpucsl &&
        python3 -m pip install -i https://test.pypi.org/simple/ gpucsl &&
        {gpucsl_cli_command("--gaussian -l 0", "coolingData")}
        """
    )

    assert_folder_exists(test_folder / output_folder)
