# kk-skeleton
Skeleton for building Kokkos and Kokkos Kernels 

1. Check out the needed `kokkos`
2. Check out the needed `kokkos-kernels`
3. Source `load-env.sh` if on a supported system

## Example Builds

baseline options on Vortex
```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_HWLOC=Off \
-DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
-DKokkosKernels_INST_DOUBLE=ON \
-DKokkosKernels_INST_FLOAT=ON \
-DKokkosKernels_INST_OFFSET_INT=OFF \
-DCMAKE_CXX_FLAGS="-lineinfo" \
-DKokkosKernels_ENABLE_TESTS=ON
```

## Additional options

* "threads" backend:
```
-DKokkos_ENABLE_THREADS=On \
-DKokkos_ENABLE_OPENMP=Off \
```

* OpenMP backend:
```
-DKokkos_ENABLE_OPENMP=On \
-DKokkos_ENABLE_THREADS=Off \
```

* CUDA on volta: 
```
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ARCH_VOLTA70=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF \
```
