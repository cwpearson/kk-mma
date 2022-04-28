project(skeleton CXX CUDA)
cmake_minimum_required(VERSION 3.16) # same as kokkos

add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Kokkos::kokkos)
target_link_libraries(main PRIVATE Kokkos::kokkoskernels)
target_compile_options(main PRIVATE ${KOKKOS_CUDA_OPTIONS})
target_link_options(main PRIVATE ${KOKKOS_LINK_OPTIONS})