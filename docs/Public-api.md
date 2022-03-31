## pc (gpucsl/pc/pc.py)

pc(
    data: np.ndarray,
    data_distribution: DataDistribution,
    max_level: int,
    alpha=0.05,
    kernels=None,
    is_debug: bool = False,
    should_log: bool = False,
    devices: List[int] = [0],
    sync_device: int = None,
    gaussian_correlation_matrix: np.ndarray = None,
) -> PCResult

Executes the PC algorithm.

### Parameters
- data: The data to analyze  
- data_distribution: Either DataDistribution.DISCRETE or DataDistribution.GAUSSIAN depending on the assumed distribution of the data  
- max_level: max level until which the pc algorithm will run (inclusive). Depending on the max level data structures will get allocated on the GPU, so you want to keep it small to avoid out of memory problems  
- alpha: Alpha value for the statistical tests
- kernels: You can compile the kernels that should be used yourself and pass them to the function. Used for time measurements where the compile time should be excluded. Leave None and GPUCSL will compile the kernels for you   
- is_debug: If set to true kernels will get compiled in debug mode
- should_log: Sets a macro 'LOG' while compiling the CUDA kernels. Can be used for custom logging from kernels  
- devices: Device IDs of GPUs to be used.  
- sync_device: Device ID of the GPU used for state synchronization in the multi GPU case (Notice: sync_device has to be in the devices list!)
  gaussian_correlation_matrix: A correlation matrix can be passed so time measurements do not inlcude the calculation. Only possible when using DataDistribution.GAUSSIAN. If None given GPUCSL calculates it itself.  

### Returns

The CPDAG that results from causal structure learning on your data, the separation sets, the maximum p values, and time measurements for the skeleton discovery and edge orientation of the pc algorithm, as well as time measurement for the execution of the kernels.

### Return Value
- PCResult

<br/><br/>
## discover_skeleton_gpu_gaussian (gpucsl/pc/discover_skeleton_gaussian.py)
discover_skeleton_gpu_gaussian(skeleton: np.ndarray,
data: np.ndarray,
correlation_matrix: np.ndarray,
alpha: float,
max_level: int,
num_variables: int,
num_observations: int,
kernels: Kernels = None,
is_debug: bool = False,
should_log: bool = False,
devices: List[int] = [0],
sync_device: int = None,) -> SkeletonResult

Performs the skeleton discovery using a conditional independence test for a multivariate normal data distribution. Offers a multi GPU support. For that, provide an array with device IDs as the devices parameter (provided your system includes multiple GPUs)

### Parameters
- skeleton: A numpy array representing a fully connected graph with as many vertices as there are variables contained in data.
- data: The original data
- correlation_matrix: The correlation matrix calculated from data
- alpha: Alpha value to do the statistical tests against
- max_level: max level until which the pc algorithm will run. Depending on the max level data will get allocated on the GPU, so you want to keep it small to avoid out of memory problems  
- num_variables: The number of variables contained in data
- num_observations: How many observations every of the variables has
- kernels:  You can compile the kernels that should be used yourself and pass them to the function. Used for time measurements where the compile time should be excluded. Leave None and GPUCSL will compile the kernels for you   
- is_debug: If set to true, kernels will get compiled in debug mode
- should_log: Sets a macro 'LOG' while compiling the CUDA kernels. Can be used for custom logging from kernels  
- devices: Device IDs of GPUs to be used.  
- sync_device: Which of the given (sync_device has to be part of devices!) should be used to sync state while using multiple gpus


### Returns
A SkeletonResult wrapped into TimedReturn. This represents the resulting undirected graph, helper structures as the separation sets (which are needed for the edge orientation), and execution time.

### Return Value
- SkeletonResult


<br/><br/>
## discover_skeleton_gpu_discrete (gpucsl/pc/discover_skeleton_dicrete.py)

discover_skeleton_gpu_discrete(
    skeleton: np.ndarray,
    data: np.ndarray,
    alpha: float,
    max_level: int,
    num_variables: int,
    num_observations: int,
    kernels=None,
    memory_restriction: int = None,
    is_debug: bool = False,
    should_log: bool = False,
) -> SkeletonResult

Does the skeleton discovery using a conditional independence test for a discrete data distribution.

### Parameters
- skeleton: A numpy array representing a fully connected graph with as many vertices as there are variables contained in data.
- data: The original data
- alpha: Alpha value to do the statistical tests against
- max_level: max level until which the pc algorithm will run. Depending on the max level data will get allocated on the GPU, so you want to keep it small to avoid out of memory problems  
- num_variables: The number of variables contained in data
- num_observations: How many observations every of the variables has
- kernels:  You can compile the kernels that should be used yourself and pass them to the function. Used for time measurements where the compile time should be   excluded. Leave None and GPUCSL will compile the kernels for you   
- memory_restriction: The maximum space to allocate for the working structures. Small values decrease the parallelisation. If None given defaults to 95% of the total available memory on GPU.
- is_debug: If set to true kernels will get compiled in debug mode
- should_log: Sets a macro 'LOG' while compiling the CUDA kernels. Can be used for custom logging from kernels  


### Returns
A SkeletonResult wrapped into TimedReturn. This represents the resulting undirected graph, helper structures as the separation sets (which are needed for the edge orientation), and execution time.

### Return Value
- SkeletonResult

<br/><br/>
## orient_edges (gpucsl/pc/edge_orientation/edge_orientation.py)

orient_edges(
    skeleton: nx.Graph,
    separation_sets: np.ndarray,
) -> nx.DiGraph

Orients the edges of the skeleton.


### Parameters
- skeleton: The result of the skeleton discovery. The undirected graph to be directed.
- separation_sets: A numpy array with the shape (num_variables, num_variables, max_level) representing the found separation sets. num_variables is the count of    vertices of the skeleton and max_level is the same that was the input the skeleton discovery function that generated the separation sets.

### Returns
The CPDAG that represents the causal dependencies

### Return Value
- networkx.DiGraph