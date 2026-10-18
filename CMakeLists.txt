cmake_minimum_required(VERSION 3.20)
project(nn_from_scratch_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(nn
    src/tensor.cpp
    src/autograd.cpp
    src/layers.cpp
    src/optimizers.cpp
    src/schedulers.cpp
    src/data.cpp
    src/trainer.cpp
)
target_include_directories(nn PUBLIC include)

function(add_nn_example name)
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE nn)
endfunction()

add_nn_example(xor_demo)
add_nn_example(spiral_demo)

enable_testing()
add_executable(nn_tests tests/tests.cpp)
target_link_libraries(nn_tests PRIVATE nn)
add_test(NAME nn_tests COMMAND nn_tests)
