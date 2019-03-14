cmake_minimum_required(VERSION 3.9.1)
project(heat2d3d)

find_package(CUDA QUIET REQUIRED)

cuda_add_executable(${PROJECT_NAME} main.cc)