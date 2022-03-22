from math import sqrt
from scipy.stats import norm
from cupyx.scipy.special import ndtr
import numpy as np
from typing import Any, Callable, Generic, NamedTuple, TypeVar, Dict, Tuple, Set
from functools import wraps
from timeit import default_timer as timer
import logging
import networkx as nx
import colorama
import cupy as cp

T = TypeVar("T")
U = TypeVar("U")

# Use for typing (cannot inherit from NamedTuple at the same time)
# keep in sync with TimedReturn
class TimedReturnT(Generic[T]):
    result: T
    runtime: float


# Use for instantiation
# keep in sync with TimedReturn
class TimedReturn(NamedTuple):
    result: Any
    runtime: float


def timed(f: Callable[..., U]) -> Callable[..., TimedReturnT[U]]:
    @wraps(f)
    def wrap(*args, **kw):
        start = timer()
        result = f(*args, **kw)
        end = timer()
        duration = end - start
        return TimedReturn(result, duration)

    return wrap


def get_gaussian_thresholds(data: np.ndarray, alpha: float = 0.05):
    num_observations = data.shape[0]
    number_of_levels = 50
    thresholds = [0.0] * number_of_levels
    for i in range(1, min(number_of_levels, num_observations - 3) - 1):
        q = norm.ppf((alpha / 2), loc=0, scale=1)
        d = sqrt(num_observations - i - 3)
        thresholds[i - 1] = abs(q / d)
    return thresholds


def init_pc_graph(data: np.ndarray):
    num_variables = data.shape[1]
    return np.ones((num_variables, num_variables), np.uint16)


def correlation_matrix_of(data: np.ndarray):
    return np.corrcoef(data, rowvar=False)


def transform_to_pmax_cupy(d_zmin: cp.ndarray, num_samples: int) -> cp.ndarray:
    # np.sqrt is only used to compute the square of a scalar
    d_intermediate_result = cp.abs(np.sqrt(num_samples - 3) * d_zmin)
    d_pmax = 2 * (1 - ndtr(d_intermediate_result))

    return d_pmax


# pmax interface should conform with pcalg pmax structure:
# lower triangle all -1
# main diagonal all -1
# all other entries are in [0, 1]
# for our tests, H_0 is "v_i and v_j are independent".
# If pvalue <= alpha, we reject H_0 and v_i and v_j are dependent (we keep the existing edge).
# If pvalue > alpha, H_0 holds and we delete the edge.
def postprocess_pmax_cupy(d_pmax: cp.ndarray) -> cp.ndarray:
    d_pmax[
        cp.tri(d_pmax.shape[0], dtype="bool")
    ] = -1  # graph also writes to lower triangle, fill this with -1

    return d_pmax


def transform_to_pmax(zmin: np.ndarray, num_samples: int):
    return 2 * norm.sf(np.abs(np.sqrt(num_samples - 3) * zmin))


# pmax interface should conform with pcalg pmax structure:
# lower triangle all -1
# main diagonal all -1
# all other entries are in [0, 1]
# for our tests, H_0 is "v_i and v_j are independent".
# If pvalue <= alpha, we reject H_0 and v_i and v_j are dependent (we keep the existing edge).
# If pvalue > alpha, H_0 holds and we delete the edge.
def postprocess_pmax(pmax: np.ndarray) -> np.ndarray:
    pmax[
        np.tril_indices(pmax.shape[0])
    ] = -1  # graph also writes to lower triangle, fill this with -1

    return pmax


def get_max_neighbor_count(
    d_compacted_skeleton: cp.ndarray, columns_per_device: int, device_index: int
) -> int:
    start_index = columns_per_device * device_index
    end_index = start_index + columns_per_device

    # If the number of devices is close to the number of the variables, some devices
    # can stay without columns assigned to them; return 0 in that case
    max_neighbors_count = (
        d_compacted_skeleton[start_index:end_index, 0].max().get()
        if d_compacted_skeleton[start_index:end_index, 0].size > 0
        else 0
    )

    return max_neighbors_count


def log(message):
    logging.info(f"{colorama.Fore.GREEN}{message}{colorama.Style.RESET_ALL}")


def log_time(message, value, unit="s", value_spacing=40):
    spacing_length = value_spacing - len(message)
    unit_message = ""
    if unit:
        spacing_length -= len(unit) + 2
        unit_message = f"({unit})"
    spacing = " " * spacing_length

    compacted_message = message + spacing + unit_message + ": " + f"{(value):1.6f}"

    log(compacted_message)


class PCResult(NamedTuple):
    directed_graph: nx.DiGraph
    separation_sets: Dict[Tuple[int, int], Set[int]]
    pmax: np.ndarray
    discover_skeleton_runtime: float
    edge_orientation_runtime: float
    discover_skeleton_kernel_runtime: float


class SkeletonResult(NamedTuple):
    skeleton: np.ndarray
    seperation_sets: Dict[Tuple[int, int], Set[int]]
    pmax: np.ndarray
    discover_skeleton_kernel_time: float
