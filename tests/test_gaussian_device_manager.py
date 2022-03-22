import pytest

from gpucsl.pc.gaussian_device_manager import GaussianDeviceManager
from gpucsl.pc.discover_skeleton_gaussian import gaussian_ci_worker
from .fixtures.input_data import Fixture, input_data
import numpy as np
import cupy as cp
import sys


global_max_level = 3
global_devices = [1, 2]
global_sync_device = 1
global_n_devices = len(global_devices)
global_device_indices = list(range(len(global_devices)))


class MockDevice(object):
    def __init__(self, device_index):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


@pytest.fixture()
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def device_manager(monkeypatch, input_data):
    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    initial_graph = input_data.fully_connected_graph
    correlation_matrix = input_data.covariance_matrix
    thresholds = input_data.thresholds
    max_level = global_max_level
    devices = global_devices

    num_observations = input_data.samples.shape[0]

    return GaussianDeviceManager(
        initial_graph,
        correlation_matrix,
        thresholds,
        num_observations,
        max_level,
        devices,
        global_sync_device,
        None,
        False,
        False,
    )


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_get_static_data(device_manager, input_data):
    (
        variable_count,
        max_level,
        thresholds,
        num_observations,
        stop_flags,
        n_devices,
        devices,
    ) = device_manager.get_static_data()

    assert variable_count == 6
    assert max_level == global_max_level
    assert np.all((thresholds == input_data.thresholds))
    assert n_devices == global_n_devices
    assert devices == global_devices
    assert stop_flags == [False for _ in range(global_n_devices)]


@pytest.mark.parametrize("device_index", global_device_indices)
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_get_data_for_device_index(device_manager, input_data, device_index):
    (
        d_skeleton,
        d_compacted_skeleton,
        d_correlation_matrix,
        d_zmin,
        d_seperation_sets,
        stream,
        ready_event,
        main_event,
        kernels,
    ) = device_manager.get_data_for_device_index(device_index)

    variable_count = 6

    assert np.all(d_skeleton.get() == input_data.fully_connected_graph)
    assert np.all(
        d_compacted_skeleton.get() == input_data.fully_connected_graph.astype(np.int32)
    )
    assert np.all(d_correlation_matrix.get() == input_data.covariance_matrix)
    assert np.all(
        d_zmin.get()
        == np.full((variable_count, variable_count), sys.float_info.max, np.float32)
    )
    assert np.all(
        d_seperation_sets.get()
        == np.full((variable_count * variable_count * global_max_level), -1, np.int32)
    )

    assert not ready_event.is_set()
    assert not main_event.is_set()

    assert cp.all(
        d_skeleton == device_manager.get_skeleton_for_device_index(device_index)
    )


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_merge_skeleton(monkeypatch, device_manager):
    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    d_skeleton1 = cp.asarray(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        cp.uint16,
    )

    d_skeleton2 = cp.asarray(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        cp.uint16,
    )

    expected_skeleton = cp.asarray(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        cp.uint16,
    )

    device_manager.d_skeletons = [d_skeleton1, d_skeleton2]
    device_manager.synchronize_skeletons()

    for d_skeleton in device_manager.d_skeletons:
        assert cp.all(d_skeleton == expected_skeleton)


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_merge_zmins(device_manager):
    d_zmin1 = cp.asarray(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        cp.float32,
    )

    d_zmin2 = cp.asarray(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        cp.float32,
    )

    expected_zmin = np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        np.float32,
    )

    device_manager.d_zmins = [d_zmin1, d_zmin2]
    d_merged_zmin = device_manager.merge_zmins()

    assert np.all(d_merged_zmin.get() == expected_zmin)


@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_merge_separation_sets(device_manager):
    # See comment in GaussianDeviceManager
    max_level = 1

    d_separation_sets1 = cp.asarray(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        cp.int32,
    ).flatten()

    d_separation_sets2 = cp.asarray(
        [
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        cp.int32,
    ).flatten()

    expected_separation_sets = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
        ],
        np.int32,
    ).flatten()

    d_mask1 = cp.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        cp.bool,
    )

    d_mask2 = cp.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        cp.bool,
    )

    device_manager.max_level = max_level
    device_manager.d_seperation_sets_array = [d_separation_sets1, d_separation_sets2]
    d_masks = [d_mask1, d_mask2]

    merged_seperation_sets = device_manager.merge_separation_sets(d_masks)

    assert np.all(expected_separation_sets == merged_seperation_sets)


def mock_worker_function(device_index, device_manager):
    (
        variable_count,
        max_level,
        thresholds,
        num_observations,
        stop_flags,
        n_devices,
        devices,
    ) = device_manager.get_static_data()

    (
        d_skeleton,
        d_compacted_skeleton,
        d_correlation_matrix,
        d_zmin,
        d_seperation_sets,
        stream,
        ready_event,
        main_event,
        kernels,
    ) = device_manager.get_data_for_device_index(device_index)

    for level in range(0, max_level + 1):
        # CI computations

        if level == 3:  # Test early termination
            stop_flags[device_index] = True
            ready_event.set()
            return

        if n_devices > 1:
            ready_event.set()
            main_event.wait()
            main_event.clear()


@pytest.mark.timeout(0.5)
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_execute_ci_workers_in_parallel_mock_worker(monkeypatch, device_manager):
    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    device_manager.execute_ci_workers_in_parallel(mock_worker_function)


@pytest.mark.timeout(1.0)
@pytest.mark.parametrize("input_data", ["coolingData"], indirect=True)
def test_execute_ci_workers_in_parallel_gaussian_worker(monkeypatch, device_manager):
    monkeypatch.setattr(cp.cuda, "Device", MockDevice)

    expected_skeleton = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        np.uint16,
    )

    final_skeleton = device_manager.execute_ci_workers_in_parallel(gaussian_ci_worker)

    assert np.all(expected_skeleton == final_skeleton)
