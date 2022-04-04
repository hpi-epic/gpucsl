# How to extend GPUCSL by a new distribution

This is a guide on how you can embed your own implementation of a CI algorithm. 

Testing your code is optional, but highly recommended. This project supports multiple types of tests:
- Python tests via pytest. They target the Python code you write.
- CUDA tests from python. They target your CUDA code callable from Python (prefixed by __global__ and with return value void).
    They enable you to prepare your data with python packages like numpy so you can make clearer what you are testing (example 
    data + call in numpy instead of randomly, externally generated input that you hardcode)
- Google test: Enables to test CUDA code not covered by the previous point or if you simply prefer to test your CUDA in C++


Now following are the necessary steps for making a new distribution accessible in GPUCSL. We recommend following the given order of steps.


## Implement your CUDA kernel

We cannot actively support you during this step, but while designing a CUDA kernel code you can reuse 
already existing helper functions and code from other distributions.

You have to place your CUDA files in the `gpucsl/pc/cuda` or `gpucsl/pc/cuda/helpers` directories! (otherwise, you have to modify
the compiler_options variable in the get_module function of `gpucsl/pc/kernel_management.py` like it is done for _cuda_dir_ or _cuda_helpers_dir_)

Note: If needed, kernels can be templated. Since the CUDA code is compiled during the runtime of GPUCSL (!) there are already some 
runtime information, like the number of variables, accessible. For more information see the next section.


## Implement your own Kernel Management

The Kernel Management encapsulates the details of kernel access, for example, the grid and block mapping of the kernel is defined here.
Otherwise encapsulates details of the underlying CuPy library (which CUDA file to read, it compiles the code on the creation of the class, function name resolution to level). 

To write your own Kernel Management you should inherit from the abstract base class Kernel. You then have to implement the following methods:

- __init__: With the __init__ method you take the parameters you need for everything involved into your Kernel Management. So if you need something to calculate your grid and block mapping you need to take these parameters here. At a minimum, you have to take the parameters is_debug and should_log.
  A template for an __init__ method is:
  ```
  def __init__(
        self,
        your_choosen_parameters,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        self.your_choosen_parameters = your_choosen_parameters

        super().__init__(is_debug=is_debug, should_log=should_log)
  ```    
- kernel_function_name_for_level: Return the name of the kernel for the given level. Can return different names for different levels (as you maybe have written 
    optimizations for a specific level in form of an extra CUDA function, as we did for DiscreteKernel. Otherwise, you can also return just a static name like the CompactKernel)
- template_specification_for_level: If you templated your kernel in the CUDA code CuPy needs the filled-out template to access the instantiated functions. Here you provide a string with the parameters you give the template for the current level.
- grid_and_block_mapping: Defines how your kernel is mapped into grids and blocks
- cuda_file: Return the filename of the cuda file containing the kernel you want to use (the path is relative to the `gpucsl/pc/cuda` directory)
- every_accessable_function_signature: Return an array containing every function signature that should be accessable later on. (You probably want use the _kernel_function_signature_for_level_ method here. It returns the kernel function signature for a level based on your _kernel_function_name_for_level_ and  _template_specification_for_level_ methods)

Optional:
- pre_kernel_launch_check: A hook executed before the raw CuPy kernel is accessed and run. Can be used to check if everything is ok before the launch happens. 
    For example: Can be used to check if the given level is below the max level.


The implemented kernel will be used in the next step.


## Implement your own skeleton discovery

Following is a template you can use for the skeleton discovery. Just copy and adapt it to your special use case. 

```
@timed
def discover_skeleton(
    skeleton: np.ndarray,
    data: np.ndarray,
    alpha: float,
    max_level: int,
    num_variables: int,
    num_observations: int,
    kernels=None,
    is_debug: bool = False,
    should_log: bool = False,
) -> SkeletonResult:

    # initialize the data structures you need on the GPU using CuPy and do your necessary setup

     d_skeleton = cp.asarray(skeleton, dtype=cp.uint16)
     d_seperation_sets = cp.full(num_variables * num_variables * max_level, -1, np.int32)
     d_pmax = cp.full((num_variables, num_variables), 0, np.float32)

    if kernels == None:
        # kernel in the singular here as in our own CI tests we always used two kernels -> plural
        # here we only assume one kernel, but you can of course use as many as you like
        kernel = YourKernel(
            your_needed_params, 
            is_debug=is_debug,
            should_log=should_log
        )

    # init stream we will need later on for synchronization
    stream = cp.cuda.get_current_stream()

    kernel_start = timer()

    for level in range(0, max_level + 1):
        # prepare whatever you need and maybe check if you are already finished with the pc algorithm
        # then execute the actual kernel, by using the kernel management you implemented earlier
        kernel(level, kernel_params) 
       

    # synchronize the stream so the times measurement is accurate and not just done later
    stream.synchronize()
    kernel_time = timer() - kernel_start

    # fetch the skeleton from the gpu
    result_skeleton = d_skeleton.get()

    # fetch separation_sets and bring them in assumed form for edge orientation
    separation_sets = d_seperation_sets.get().reshape(
        (num_variables, num_variables, max_level)
    )

    # bring the pmax values in the same form pcalg would return
    pmax = postprocess_pmax(d_pmax.get())

    return SkeletonResult(result_skeleton, separation_sets, pmax, kernel_time)
```

Note: make sure the types you initialize your data with on Python side match the data types the CUDA kernel takes. An example in the template is d_skeleton which is initialized as cp.uint16.

After implementing the kernel discovery you inherit from the abstract _PC_(`gpucsl/pc/pc.py`) class. Now implement the methods:
-  _set_distribution_specific_options_ method: Takes the arguments specific to your distribution and saves them as instance variables. You can return the object itself in the end in order to chain the execute call to it easier.
- _discover_skeleton_: Returns the result of the earlier implemented skeleton discovery. You can just pass the locally saved parameters to your earlier implemented _discover_skeleton_ function or implement the complete skeleton discovery here and just use the instance variables directly


## Extend CLI (optional) 

Depending on how you want to use GPUCSL you want to extend the CLI (if you just want to call it from Python you can ignore this step).
You mainly have to extend the command line parser (gpucsl/pc/command_line_parser.py) with the arguments you need. The parser uses the argparse package, so please refer to https://docs.python.org/3/library/argparse.html on how to use it.
basically used the argparse package, so please refer to https://docs.python.org/3/library/argparse.html on how to use it.

One mandatory step will be to add a new distribution flag (example for distribution flag: gaussian). Then you have to extend the function _gpucsl_cli_ 
(`gpucsl/cli/cli.py`). The main points are error checking for new parameters you introduced, instanciate your implemented subclass of _PC_ and pass the sanitized values to the class by creating it with the general parameters the constructor takes and calling the _set_distribution_specific_options_ method with the distribution specific options as arguments. 
You do not need to change anything to write the results as done currently.
