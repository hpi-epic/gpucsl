#!/bin/bash

# It is only possible to include and test files that do not contain
# device specific code like threadIdx.x or atomicAdd(...), as these can not
# be provided for these tests, as they are run on the CPU
# If one includes these files, they need to be hidden to the compiler, e.g. by
# preprocessor macros


PS4="\n\033[1;33m>>>\033[0m "; set -x

cd tests/cuda/ && cmake -S . -B build && cmake --build build && cd build && ctest -V
