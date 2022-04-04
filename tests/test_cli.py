from unittest.mock import MagicMock
from pytest_mock import MockerFixture
import pytest
from gpucsl.cli.cli import gpucsl_cli as rcsl
from gpucsl.cli.cli_io import force_csv_suffix
import argparse
from pathlib import Path
import logging
import os
import cupy as cp
from gpucsl.cli.cli_util import GpucslError
from gpucsl.cli.cli_io import (
    force_csv_suffix,
    get_file_name_stem,
    read_correlation_matrix,
    write_results,
)


# mock setup
graph_data = "graph data"
separation_sets = {(0, 0): {5}}


class ExceptionForTests(Exception):
    pass


def exit(*params):
    raise ExceptionForTests()


class MockShape:
    shape = [0, 15]


class MockClass:
    def to_numpy(*params):
        return MockShape()


def read_csv(path, header):
    return MockClass()


class MockInnerResult:
    directed_graph = graph_data
    separation_sets = separation_sets
    pmax = None


class MockResult:
    result = MockInnerResult()


class MockPC:
    def set_distribution_specific_options(self, *args, **kwargs):
        pass

    def execute(self):
        return MockResult()


def pc(*args, **kwargs):
    return MockPC()


@pytest.fixture
def setup(mocker):
    mocker.patch.object(argparse.ArgumentParser, "exit", exit)
    mocker.patch("os.makedirs")
    mocker.patch("pandas.read_csv", read_csv)
    mocker.patch("gpucsl.cli.cli.GaussianPC", pc)
    mocker.patch("gpucsl.cli.cli.DiscretePC", pc)
    mocker.patch("gpucsl.cli.cli.write_results_and_config")
    mocker.patch("numpy.savetxt")


# Tests
def test_error_on_no_parameter(setup):
    with pytest.raises(ExceptionForTests):
        rcsl([])


def test_error_on_missing_dataset(setup):
    with pytest.raises(ExceptionForTests):
        rcsl(["-l", "3", "-o", "foo/bar/foobar", "--gaussian"])


def test_error_on_missing_output_path(setup):
    with pytest.raises(ExceptionForTests):
        rcsl(["-d", "foobar", "-l", "3", "--gaussian"])


def test_either_discrete_or_gaussian_has_to_be_set(setup):
    with pytest.raises(ExceptionForTests):
        rcsl(["-d", "foobar", "-l", "3", "--gaussian"])


def test_discrete_and_gaussian_cannot_be_set(setup):
    with pytest.raises(ExceptionForTests):
        rcsl(["-d", "foobar", "-l", "3", "--gaussian", "--discrete"])


def test_error_on_discrete_and_correlation_matrix_given(setup):
    with pytest.raises(GpucslError):
        rcsl(
            [
                "-d",
                "foobar",
                "-o",
                "foo/bar/foobar",
                "-l",
                "3",
                "--discrete",
                "-c",
                "correlation//matrix/path",
            ]
        )


def test_error_on_discrete_and_multiple_gpus_given(setup):
    with pytest.raises(GpucslError):
        rcsl(["-d", "foobar", "-o", "foo/bar/foobar", "--discrete", "-g", "0", "1"])


def test_alpha_has_to_be_between_0_and_1(setup):
    with pytest.raises(ExceptionForTests):
        rcsl(["-d", "foobar", "-l", "3", "--gaussian", "-a", "-1"])
    with pytest.raises(ExceptionForTests):
        rcsl(["-d", "foobar", "-l", "3", "--gaussian", "-a", "1.1"])


def test_csv_suffix_is_preserved():
    path = Path("foobar/foo.csv")

    with_forced_suffix = force_csv_suffix(path)

    assert path == with_forced_suffix


def test_csv_suffix_is_added():
    path_string = "foobar/foo"
    path = Path(path_string)

    with_forced_suffix = force_csv_suffix(path)

    assert Path(path_string + ".csv") == with_forced_suffix


def test_dataset_name_gets_correctly_determined():
    file_name = "foo"
    path = f"foobar/{file_name}.csv"

    dataset_name = get_file_name_stem(path)

    assert dataset_name == file_name


def test_read_correlation_martrix_preserves_none():
    assert read_correlation_matrix(None) == None


def test_alpha_gets_set(setup, mocker):
    alpha = 0

    def pc(data, max_level, alph, **kwargs):
        nonlocal alpha
        alpha = alph

        return MockPC()

    mocker.patch("gpucsl.cli.cli.GaussianPC", pc)

    expected_alpha = 0.6

    rcsl(
        [
            "-d",
            "foobar",
            "--gaussian",
            "-l",
            "3",
            "-o",
            "foo/bar/foobar",
            "-a",
            str(expected_alpha),
        ]
    )
    assert alpha == expected_alpha


def test_max_level_gets_set(setup, mocker):
    max_level = 0

    def pc(data, level, alpha, **kwargs):
        nonlocal max_level
        max_level = level
        print(data, level)

        return MockPC()

    mocker.patch("gpucsl.cli.cli.GaussianPC", pc)

    expected_max_level = 5

    rcsl(
        [
            "-d",
            "foobar",
            "--gaussian",
            "-l",
            str(expected_max_level),
            "-o",
            "foo/bar/foobar",
        ]
    )
    assert max_level == expected_max_level


def test_no_log_level_gets_set(setup, mocker):
    log_level = None

    def log(level):
        nonlocal log_level
        log_level = level

        return MockResult()

    mocker.patch("logging.basicConfig", log)
    rcsl(["-d", "foobar", "--gaussian", "-l", "2", "-o", "foo/bar/foobar"])

    assert log_level is None


def test_log_level_1_gets_set(setup, mocker):
    log_level = None

    def log(level):
        nonlocal log_level
        log_level = level

        return MockResult()

    mocker.patch("logging.basicConfig", log)
    rcsl(["-d", "foobar", "--gaussian", "-l", "2", "-o", "foo/bar/foobar", "-v"])

    assert log_level == logging.INFO


def test_log_level_2_gets_set(setup, mocker):
    log_level = None

    def log(level):
        nonlocal log_level
        log_level = level

        return MockResult()

    mocker.patch("logging.basicConfig", log)
    rcsl(["-d", "foobar", "--gaussian", "-l", "2", "-o", "foo/bar/foobar", "-vv"])

    assert log_level == logging.DEBUG


def test_directory_gets_created_when_non_existing(setup, mocker):
    dir = ""

    def mkdirs(path):
        nonlocal dir
        dir = path

    # mocker.patch("os.path.isdir", result=True)
    mocker.patch("os.makedirs", mkdirs)

    mocker.patch("gpucsl.cli.cli_io.write_directed_graph")
    mocker.patch("gpucsl.cli.cli_io.write_seperation_set")
    mocker.patch("gpucsl.cli.cli_io.write_pmax")

    to_be_created_path = "path/to/be/created"

    write_results(to_be_created_path, "file_name", MagicMock())

    assert dir == to_be_created_path


def test_directory_gets_not_created_when_existing(setup, mocker):
    dir = None

    def mkdirs(path):
        nonlocal dir
        dir = path

    mocker.patch("os.path.isdir", result=True)
    mocker.patch("os.makedirs", mkdirs)

    mocker.patch("gpucsl.cli.cli_io.write_directed_graph")
    mocker.patch("gpucsl.cli.cli_io.write_seperation_set")
    mocker.patch("gpucsl.cli.cli_io.write_pmax")

    write_results("path/to/be/created", "file_name", MagicMock())

    assert dir is None


def test_read_csv_from_given_path(setup, mocker):
    path = None

    def read_csv(pa, header):
        nonlocal path
        path = pa

        return MockClass()

    mocker.patch("pandas.read_csv", read_csv)

    file_name = "foobar"

    rcsl(["-d", file_name, "--gaussian", "-l", "3", "-o", "foo/bar/foobar"])

    assert path == Path(f"{os.getcwd()}/{file_name}.csv")
