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

This is the part we can help you with the least. You have to design efficient CUDA kernel code. While doing so you can use 
any of the already existing helper functions and you can reuse code from our distributions however you like.

You have to place your CUDA files in the gpucsl/pc/cuda or gpucsl/pc/cuda/helpers directories! (otherwise, you have to modify
the compiler_options variable in the get_module function of gpucsl/pc/kernel_management.py like it is done for cuda_dir or cuda_helpers_dir)

Note: If needed, kernels can be templated. Since the CUDA code is compiled during the runtime of GPUCSL (!) there are already some 
runtime information, like the number of variables, accessible. For more information see the next section.


## Implement your own Kernel Management

The Kernel Management encapsulates the details of kernel access, for example, the grid and block mapping of the kernel is defined here.
Otherwise encapsulates details of the underlying CuPy library (which CUDA file to read, it compiles the code on the creation of the class, function name resolution to level). 

To write your own Kernel Management you want to inherit from the abstract base class Kernel. You then have to implement the following methods:

- __init__: With the __init__ method you take the paramters you need for everything involved into your kernel management. So if you need something to calculate your grid and block mapping you need to take this parameters here. At a minimum you have to take the paramters is_debug and should_log.
  A template for an __init__ method is:
  ```
  def __init__(
        self,
        your_choosen_parameters,
        is_debug: bool = False,
        should_log: bool = False,
    ):
        super().__init__(is_debug=is_debug, should_log=should_log)
        self.your_choosen_parameters = your_choosen_parameters

        # calculate all the kernel function signature names you later on want to be able to call (something needed by cupy)
        # has to be an array, but if you only need one signature you can change this to be an array with one entry
        kernel_function_signatures = [
            self.kernel_function_signature_for_level(level)
            for level in range(0, max_level + 1)
        ]

        # pass the kernel function signatures and compile the cuda code
        self.define_module("gaussian_ci.cu", kernel_function_signatures)
  ```    
- kernel_function_name_for_level: Return the name of the kernel for the given level. Can return different names for different levels (as you maybe have written 
    optimizations for a specific level in form of an extra CUDA function, like we did for DiscreteKernel. Otherwise you can also return just a static name like the CompactKernel)
- template_specification_for_level: If you templated your kernel in the CUDA code CuPy needs the filled-out template to access the instantiated functions. Here you provide a string with the parameters you give the template for the current level.
- grid_and_block_mapping: Defines how your kernel is mapped into grids and blocks

Optional:
- pre_kernel_launch_check: A hook executed before the raw CuPy kernel is accessed and run. Can be used to check if everything is ok before the launch happens. 
    For example: Can be used to check if the given level is below the max level.


The implemented kernel will be used in the next step.


## Implement your own skeleton discovery

Following is a template you can use for the skeleton discovery. Just copy and adapt it to your special use case. 

```
@timed
def discover_skeleton_gpu_discrete(
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

Note: make sure the types you initialize your data with on python side matches the data types the CUDA kernel takes. An example in the template is d_skeleton that is initialized as cp.uint16.  

After implementing the kernel discovery you can extend the pc function (gpucsl/pc/pc.py). First add your distribution to the DataDistribution enum in the same file. Then in pc test for your DataDistribution and call your skeleton discovery like it is done for DataDistribution.GAUSSIAN or DataDistribution.DISCRETE. Depending on whether you need more arguments you have to extend the pc functions argument list. 
As long as you return a correct SkeletonResult from your skeleton discovery you do not need to change anything else. The edge orientation should work.


## Extend CLI (optional) 

Depending on how you want to use GPUCSL you want to extend the CLI (if you just want to call it from python you can ignore this step).
You mainly have to extend the command line parser (gpucsl/pc/command_line_parser.py) with the arguments you need. The parser 
basically used the argparse package, so please refer to https://docs.python.org/3/library/argparse.html on how to use it.

One mandatory step will be to add a new distribution flag (example for distribution flag: gaussian). Then you have to extend the function `gpucsl_cli` 
(gpucsl/cli/cli.py). The main points are error checking for new parameters you introduced and passing the sanitized values to the run_on_dataset function. 
You do not need to change anything to write the results as done currently.
