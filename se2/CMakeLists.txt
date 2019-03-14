cmake_minimum_required(VERSION 3.9.1)
project(heat1D)

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
include_directories(
    src
)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} src/main.cc)