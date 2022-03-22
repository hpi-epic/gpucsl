from gpucsl.pc.kernel_management import CompactKernel, GaussianKernel, DiscreteKernel
from timeit import default_timer as timer

import pytest


@pytest.fixture
def gaussian_kernel_getters():
    return GaussianKernel(6, 1, 1)


@pytest.fixture
def discrete_kernel_getters():
    return DiscreteKernel(1, 1, 1, 1)


@pytest.fixture
def compact_kernel_getter():
    return CompactKernel(1, 1)


def test_get_kernel_module(
    gaussian_kernel_getters, discrete_kernel_getters, compact_kernel_getter
):
    get_gaussian_kernel = gaussian_kernel_getters.get_function_for_level
    get_compact_kernel = compact_kernel_getter.get_function_for_level
    get_discrete_kernel = discrete_kernel_getters.get_function_for_level

    kernels = [
        get_gaussian_kernel(0),
        get_gaussian_kernel(1),
        get_compact_kernel(0),
        get_discrete_kernel(0),
        get_discrete_kernel(1),
    ]

    for kernel in kernels:
        assert kernel is not None


def test_cannot_execute_greater_ci_kernel_than_max_level(
    gaussian_kernel_getters, discrete_kernel_getters
):
    get_gaussian_kernel = gaussian_kernel_getters
    get_discrete_kernel = discrete_kernel_getters

    with pytest.raises(AssertionError):
        get_gaussian_kernel(2)

    with pytest.raises(AssertionError):
        get_discrete_kernel(2)


def test_kernel_caching():

    total_start = timer()

    # This is just a random number that we need to initialize the kernels
    magic_kernel_init_param = 4

    start = timer()
    gaussian_kernel = GaussianKernel(
        magic_kernel_init_param, 1, magic_kernel_init_param
    )
    compact_kernel = CompactKernel(magic_kernel_init_param, 1)
    discrete_kernel = DiscreteKernel(
        magic_kernel_init_param,
        magic_kernel_init_param,
        magic_kernel_init_param,
        magic_kernel_init_param,
    )

    init_duration = timer() - start
    print(f"init duration: {init_duration}")

    level_to_load = 2

    start = timer()
    gaussian_kernel.get_function_for_level(level_to_load)
    compact_kernel.get_function_for_level(0)
    discrete_kernel.get_function_for_level(level_to_load)
    first_get_duration = timer() - start
    print(f"first get duration: {first_get_duration}")

    assert init_duration > (0.99 * first_get_duration)
    # Test for kernel function caching. Above test shows that
    # more than 99% of the time is spent in the initialization
    # of the Kernels to compile the cupy module. The other calls
    # just access the cached cupy kernels.


@pytest.mark.parametrize("memory_restriction", [100_000_000, 500_000_000])
def test_discrete_kernel_configuration_is_valid_for_memory_restriction(
    memory_restriction,
):

    max_level = 4
    variable_count = 10
    max_dim = 10
    n_observations = 100

    discrete_kernel = DiscreteKernel(
        max_level,
        variable_count,
        max_dim,
        n_observations,
        memory_restriction=memory_restriction,
    )
    assert discrete_kernel.max_memory_size < memory_restriction


def test_discrete_kernel_fails_if_not_enough_memory():

    # This setup can not give enough memory,
    # as alone one contingency table would be 100^100
    max_level = 100
    variable_count = 100
    max_dim = 100
    n_observations = 100

    with pytest.raises(RuntimeError):
        _ = DiscreteKernel(
            max_level,
            variable_count,
            max_dim,
            n_observations,
        )
