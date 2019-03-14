cmake_minimum_required(VERSION 3.8)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(THREADS_PREFER_PTHREAD_FLAG ON)

include_directories(${MPI_C_INCLUDE_PATH})
include(FindCUDA)

include_directories(
	src
)

set(SOURCE_FILES
	src/complex.h
	src/complex.cc
	src/input_image.h
	src/input_image.cc
)

add_executable(p31 ${SOURCE_FILES} main_thread.cc)
add_executable(p32 ${SOURCE_FILES} main_mpi.cc)
cuda_add_executable(p33 ${SOURCE_FILES} main_cuda.cu)

target_link_libraries (p31 Threads::Threads ${MPI_LIBRARIES})
target_link_libraries (p32 Threads::Threads ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(p32 PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(p32 PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()