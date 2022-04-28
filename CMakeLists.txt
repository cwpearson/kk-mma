project(skeleton CXX CUDA)
cmake_minimum_required(VERSION 3.16) # same as kokkos

add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)