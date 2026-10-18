cmake_minimum_required(VERSION 3.20)
project(nn_from_scratch_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Library, examples, and tests land here as the framework is built.
enable_testing()
