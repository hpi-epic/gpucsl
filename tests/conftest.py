import numpy as np
import pytest
import colorama


def pytest_addoption(parser):
    parser.addoption(
        "--run_testPyPI",
        action="store_true",
        help="run the tests only in case of that command line (marked with marker @run_testPyPI)",
    )
    parser.addoption(
        "--run_slow",
        action="store_true",
        help="run slow tests only in case of that command line (marked with marker @run_slow)",
    )


def pytest_runtest_setup(item):
    if "run_testPyPI" in item.keywords and not item.config.getoption("--run_testPyPI"):
        pytest.skip("need --run_testPyPI option to run this test")
    if "run_slow" in item.keywords and not item.config.getoption("--run_slow"):
        pytest.skip("need --run_slow option to run this test")


np.set_printoptions(
    formatter={
        "bool": lambda x: f"{colorama.Fore.GREEN if x != 0 else colorama.Fore.RED}{x}{colorama.Fore.BLACK}",
        "int": lambda x: f"{colorama.Fore.GREEN if x != 0 else colorama.Fore.BLACK}{x}{colorama.Fore.BLACK}",
    },
    linewidth=300,
    threshold=1000,
    precision=3,
)
