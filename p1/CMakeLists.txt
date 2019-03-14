cmake_minimum_required(VERSION 3.9.0)
project(p1)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(p1
        src/array.cc
        src/array.h
        src/simple_string.cc
        src/simple_string.h
        main.cc)
