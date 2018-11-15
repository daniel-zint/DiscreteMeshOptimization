cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(DMO LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenMesh
find_package(OpenMesh REQUIRED)
include_directories(${OPENMESH_INCLUDE_DIRS})
add_definitions(-D_USE_MATH_DEFINES)

# Find CUDA
file(GLOB SRC src/*.cpp src/*.cu)
file(GLOB HEADER src/*.h)
# Header only Cuda Library
file(GLOB CUDA_LIB std/cudalib/*.h)
message(STATUS "Files: ${SRC} ${HEADER}")

add_executable(DMO ${SRC} ${HEADER} ${CUDA_LIB})
target_link_libraries(DMO ${OPENMESH_LIBRARIES})
set_target_properties(DMO PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")