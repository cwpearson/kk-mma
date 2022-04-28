#! /bin/bash

host=`hostname`

echo `dirname $BASH_SOURCE`
DIR=`dirname ${BASH_SOURCE}`

KOKKOS_SRC="$DIR/kokkos"
KERNELS_SRC="$DIR/kokkos-kernels"
KOKKOS_BUILD="$DIR/kokkos"
KERNELS_BUILD="$DIR/kokkos"
NVCC_WRAPPER=`readlink -f "${KOKKOS_SRC}/bin/nvcc_wrapper"`

if [[ "$NERSC_HOST" == perlmutter ]]; then
    # compiler segfault with gcc/10.3.0

    echo \$NERSC_HOST matched perlmutter

    export CUDAARCHS="80" # for cmake 3.20+
    module load cuda/11.3.0
    module load cmake/3.20.5
    module load gcc/9.3.0

    which cmake
    which gcc
    which nvcc
elif [[ "$host" =~ .*ascicgpu.* ]]; then
    echo "$host" matched ascicgpu
    
    export CUDAARCHS="70" # for cmake 3.20+

    module purge

    mkdir -p /tmp/$USER
    export TMPDIR=/tmp/$USER

    module load sierra-devel/nvidia

    module load cde/v2/cmake/3.19.2

    which cmake
    which gcc
    which nvcc
    which mpirun
elif [[ "$host" =~ .*vortex.* ]]; then
# CUDA 10.1 & cmake 3.18.0 together cause some problem with recognizing the `-pthread` flag.

    echo "$host" matched vortex
    
    echo "export CUDAARCHS=70"
    export CUDAARCHS="70" # for cmake 3.20+

    echo "export OMPI_CXX=${NVCC_WRAPPER}"
    export OMPI_CXX="${NVCC_WRAPPER}"

    echo module --force purge
    module --force purge

    echo module load cmake/3.18.0
    module load cmake/3.18.0
    echo module load cuda/10.2.89
    module load cuda/10.2.89
    echo module load gcc/7.3.1
    module load gcc/7.3.1
    echo module load spectrum-mpi/rolling-release
    module load spectrum-mpi/rolling-release
    echo module load nsight-compute/2020.3.1
    module load nsight-compute/2020.3.1

    which cmake
    which gcc
    which nvcc
    which mpirun
fi
