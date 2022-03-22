cmake_minimum_required(VERSION 3.14)
project(gpucsl_cuda_tests)

# GoogleTest requires at least C++11
set(CMAKE_CXX_STANDARD 11)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  helper_test
  helper_test.cc
)
target_link_libraries(
  helper_test
  gtest_main
)

add_compile_definitions(GTEST)

include(GoogleTest)
gtest_discover_tests(helper_test)
