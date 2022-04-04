from math import ceil, floor
from typing import Tuple, Callable, List
import cupy as cp
from pathlib import Path
import math
from abc import ABC, abstractmethod

from gpucsl.pc.helpers import log

current_dir = Path(__file__).parent
cuda_dir = current_dir / "cuda"
cuda_helpers_dir = cuda_dir / "helpers"


def get_module(
    file_name: str,
    name_expressions,
    additional_compiler_options: tuple = (),
    is_debug: bool = False,
    should_log: bool = False,
):
    with open(cuda_dir / file_name) as f:
        loaded_from_source = f.read()

    compiler_options = (
        (
            "--std=c++11",
            "-I%s" % (cuda_dir),
            "-I%s" % (cuda_helpers_dir),
        )
        + additional_compiler_options
        + (("-G", "-lineinfo") if is_debug else ())
        + (("-D", "LOG") if should_log else ())
    )

    module = cp.RawModule(
        code=loaded_from_source,
        jitify=True,
        name_expressions=name_expressions,
        options=compiler_options,
    )

    return module


def get_blocks_threads(variable_count: int):
    threads = 32  # 32 * 32 = 1024
    blocks = ceil(variable_count / threads)

    return ((blocks, blocks), (threads, threads))


def calculate_blocks_and_threads_kernel_level_0(variable_count: int, n_devices: int):
    needed_threads = floor((variable_count * (variable_count + 1)) / 2)
    blocks_per_grid = (ceil(needed_threads / 1024),)
    threads_per_block = (min(max(32, needed_threads), 1024),)

    return (blocks_per_grid, threads_per_block)


def calculate_blocks_and_threads_kernel_level_n(variable_count: int, n_devices: int):
    blocks_per_grid = (int(ceil(variable_count / n_devices)), variable_count)
    threads_per_block = (32,)

    return (blocks_per_grid, threads_per_block)


def calculate_blocks_and_threads_compact(variable_count: int, n_devices: int):
    columns_per_device = int(ceil(variable_count / n_devices))
    blocks_per_grid = (int(ceil(columns_per_device / 512)),)
    threads_per_block = (min(512, columns_per_device),)

    return (blocks_per_grid, threads_per_block)


KernelLauncher = Callable[..., None]
KernelGetter = Callable[[int], KernelLauncher]


def make_kernel_launcher(
    module,
    kernel_function_name: str,
    template_specification: str,
    blocks_per_grid: tuple,
    threads_per_block: tuple,
) -> KernelLauncher:
    kernel = module.get_function(kernel_function_name + template_specification)

    return lambda parameters: kernel(blocks_per_grid, threads_per_block, parameters)


class Kernel(ABC):
    def __init__(self, is_debug: bool = False, should_log: bool = False):
        self.is_debug = is_debug
        self.should_log = should_log

        kernel_function_signatures = self.every_accessable_function_signature()

        self.define_module(self.cuda_file(), kernel_function_signatures)

    def define_module(self, cuda_file_name: str, name_expressions):
        self.module = get_module(
            cuda_file_name,
            name_expressions,
            is_debug=self.is_debug,
            should_log=self.should_log,
        )

        # compile the module explicitly, so we know when the compilation happens
        self.module.compile()

    @abstractmethod
    def cuda_file(self) -> str:
        pass

    @abstractmethod
    def kernel_function_name_for_level(self, level: int) -> str:
        pass

    @abstractmethod
    def template_specification_for_level(self, level: int) -> str:
        pass

    def kernel_function_signature_for_level(self, level) -> str:
        return f"{self.kernel_function_name_for_level(level)}{self.template_specification_for_level(level)}"

    @abstractmethod
    def every_accessable_function_signature(self) -> List[str]:
        pass

    @abstractmethod
    def grid_and_block_mapping(self, level) -> Tuple[Tuple, Tuple]:
        pass

    def get_function_for_level(self, level: int):
        return self.module.get_function(self.kernel_function_signature_for_level(level))

    def launch(self, level: int, parameters: Tuple):
        kernel = self.get_function_for_level(level)

        blocks_per_grid, threads_per_block = self.grid_and_block_mapping(level)

        kernel(blocks_per_grid, threads_per_block, parameters)

    # Hook before the kernel launch is executed. Here you can check if everything is ok before your kernel is launched
    def pre_kernel_launch_check(self, level):
        pass

    def __call__(self, level, *parameters):
        self.pre_kernel_launch_check(level)
        self.launch(level, parameters)


class CompactKernel(Kernel):
    def __init__(
        self,
        variable_count,
        device_count,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        self.device_count = device_count
        self.variable_count = variable_count
        self.columns_per_device = int(ceil(variable_count / self.device_count))

        super().__init__(is_debug=is_debug, should_log=should_log)

    def cuda_file(self) -> str:
        return "helpers/graph_helpers.cu"

    def kernel_function_name_for_level(self, level: int) -> str:
        return "compact"

    def template_specification_for_level(self, level: int) -> str:
        return f"<{self.variable_count}, {self.columns_per_device}>"

    def every_accessable_function_signature(self) -> List[str]:
        # given level does not matter as it is only for interface compliance
        return [self.kernel_function_signature_for_level(0)]

    # horizontal partitioning of adjacency matrix per device
    def grid_and_block_mapping(self, level: int):
        columns_per_device = int(ceil(self.variable_count / self.device_count))
        blocks_per_grid = (int(ceil(columns_per_device / 512)),)
        threads_per_block = (min(512, columns_per_device),)

        return (blocks_per_grid, threads_per_block)


class GaussianKernel(Kernel):
    def __init__(
        self,
        variable_count: int,
        device_count: int,
        max_level: int,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        self.variable_count = variable_count
        self.device_count = device_count
        self.max_level = max_level

        super().__init__(is_debug=is_debug, should_log=should_log)

    def cuda_file(self) -> str:
        return "gaussian_ci.cu"

    def kernel_function_name_for_level(self, level: int) -> str:
        return "gaussian_ci_level_0" if level == 0 else "gaussian_ci_level_n"

    def template_specification_for_level(self, level: int) -> str:
        return f"<{level}, {self.variable_count}, {self.max_level}>"

    def every_accessable_function_signature(self) -> List[str]:
        return [
            self.kernel_function_signature_for_level(level)
            for level in range(0, self.max_level + 1)
        ]

    def grid_and_block_mapping(self, level: int):
        mapping = None
        if level == 0:
            mapping = calculate_blocks_and_threads_kernel_level_0
        else:
            mapping = calculate_blocks_and_threads_kernel_level_n

        return mapping(self.variable_count, self.device_count)

    def pre_kernel_launch_check(self, level):
        assert level <= self.max_level


class DiscreteKernel(Kernel):
    def __init__(
        self,
        max_level: int,
        variable_count: int,
        max_dim: int,
        n_observations: int,
        memory_restriction: int = None,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        self.max_level = max_level
        self.variable_count = variable_count
        self.max_dim = max_dim
        self.n_observations = n_observations

        (
            self.blocks_per_grid_per_level,
            self.threads_per_block_per_level,
            self.max_memory_size,
            self.parallel_ci_tests_in_block,
        ) = find_execution_specification(
            variable_count, max_level, max_dim, memory_restriction=memory_restriction
        )

        super().__init__(is_debug=is_debug, should_log=should_log)

    def cuda_file(self) -> str:
        return "discrete_ci.cu"

    def kernel_function_name_for_level(self, level: int) -> str:
        return "discrete_ci_level_0" if level == 0 else "discrete_ci_level_n"

    def template_specification_for_level(self, level: int) -> str:
        return f"<{level}, {self.variable_count}, {self.parallel_ci_tests_in_block}, {self.max_dim}, {self.n_observations}, {self.max_level}>"

    def every_accessable_function_signature(self) -> List[str]:
        return [
            self.kernel_function_signature_for_level(level)
            for level in range(0, self.max_level + 1)
        ]

    def grid_and_block_mapping(self, level: int):
        if level == 0:
            blocks_per_grid, threads_per_block = (
                (self.variable_count, self.variable_count),
                (32, 1),
            )
        else:
            blocks_per_grid, threads_per_block = (
                self.blocks_per_grid_per_level[level],
                self.threads_per_block_per_level[level],
            )

        return blocks_per_grid, threads_per_block

    def pre_kernel_launch_check(self, level):
        assert level <= self.max_level


class Kernels:
    def __init__(self):
        self.ci_test = None
        self.compact = None

    @staticmethod
    def for_gaussian_ci(
        variable_count: int,
        device_count: int,
        max_level: int,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        instance = Kernels()

        instance.ci_test = GaussianKernel(
            variable_count, device_count, max_level, is_debug, should_log
        )

        instance.compact = CompactKernel(
            variable_count, device_count, is_debug, should_log
        )

        return instance

    @staticmethod
    def for_discrete_ci(
        max_level: int,
        variable_count: int,
        max_dim: int,
        n_observations: int,
        memory_restriction: int = None,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        instance = Kernels()

        instance.ci_test = DiscreteKernel(
            max_level,
            variable_count,
            max_dim,
            n_observations,
            memory_restriction,
            is_debug,
            should_log,
        )

        instance.compact = CompactKernel(variable_count, 1, is_debug, should_log)

        return instance


def contingency_and_marginal_storage_size(level: int, max_dim: int) -> int:
    maximum_dimension_of_s = math.pow(max_dim, level)
    # Account for storages of contingency table and three marginal tables
    return (
        maximum_dimension_of_s * max_dim * max_dim
        + 2 * maximum_dimension_of_s * max_dim
        + maximum_dimension_of_s
    )


def parallel_executed_ci_tests(
    blocks_per_grid: Tuple[int, int], threads_per_block: Tuple[int, int]
) -> int:
    return (
        blocks_per_grid[0]
        * blocks_per_grid[1]
        * threads_per_block[1]
        * threads_per_block[2]
    )


def bytes_to_giga_bytes(bytes: int):
    return bytes / 1_000_000_000


# Tries to find the optimal execution specification
# This is different for different datasets, so it is necessary to use heuristics.
#
# For each level, we check if the current parallelization is able to be executed with
# the given memory. For each ci-test that can be run in parallel, we need a unique
# memory space for the contingency and marginal matrix (see `calculate_working_memory`).
# If too much memory would be needed, we reduce the parallelism and execute some test
# sequencially, which allows us to reuse the memory, and use less memory.
#
# In detail, we reduce the number of blocks, by pulling memory_reduction_factor work
# blocks together in one work block, which then executes these memory_reduction_factor
# times work blocks sequentially.
def find_execution_specification(
    variable_count: int, max_level: int, max_dim: int, memory_restriction=None
):

    blocks_per_grid_per_level = {}
    threads_per_block_per_level = {}

    memory_reduction_factor = 1

    if memory_restriction is None:
        # Check free memory on device:
        free_memory, total_memory = cp.cuda.runtime.memGetInfo()
        print(
            f"Total available memory on device: {bytes_to_giga_bytes(total_memory)}GB"
        )
        print(f"Free memory on device: {bytes_to_giga_bytes(free_memory)}GB")
        memory_restriction = free_memory * 0.95

    max_memory_size = 0

    level = 1

    while level <= max_level:
        parallel_ci_worker_threads = 64
        parallel_ci_tests_per_edge_in_block = 2
        parallel_edges_in_block = 1

        blocks_per_grid = (
            math.ceil(variable_count / memory_reduction_factor),
            math.ceil(variable_count / parallel_edges_in_block),
        )

        threads_per_block = (
            parallel_ci_worker_threads,
            parallel_ci_tests_per_edge_in_block,
            parallel_edges_in_block,
        )

        memory_size = calculate_working_memory_size(
            level,
            max_dim,
            blocks_per_grid,
            threads_per_block,
        )

        if memory_size > memory_restriction:
            # Retry with more memory reduction
            memory_reduction_factor += 1
            log(
                f"Needed to reduce parallelization in order to use less memory. Current memory reduction factor is {memory_reduction_factor}"
            )

            if memory_reduction_factor > variable_count:
                log(
                    "Did not find a valid configuration that uses a small amount of memory"
                )
                log(
                    f"With maximum memory reduction factor, we still need {bytes_to_giga_bytes(memory_size)}GB space, but have given only {bytes_to_giga_bytes(memory_restriction)}GB"
                )
                log(f"Please reduce the maximum level to {level - 1}")
                raise RuntimeError(f"Out of memory for maximum level of {level}")
            continue

        # Store configuration for level
        max_memory_size = max(max_memory_size, memory_size)
        blocks_per_grid_per_level[level] = blocks_per_grid
        threads_per_block_per_level[level] = threads_per_block
        level += 1

    return (
        blocks_per_grid_per_level,
        threads_per_block_per_level,
        max_memory_size,
        parallel_ci_tests_per_edge_in_block * parallel_edges_in_block,
    )


def calculate_working_memory_size(level, max_dim, blocks_per_grid, threads_per_block):
    sizeof_uint32 = 4
    return (
        sizeof_uint32
        * contingency_and_marginal_storage_size(level, max_dim)
        * parallel_executed_ci_tests(blocks_per_grid, threads_per_block)
    )
