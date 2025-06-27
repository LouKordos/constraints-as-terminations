cmake_minimum_required(VERSION 3.28.0)
project(cat_sim2real)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Generate json file for VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)