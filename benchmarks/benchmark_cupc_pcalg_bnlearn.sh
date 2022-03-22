#!/bin/bash
PS4="\n\033[1;33m>>>\033[0m "; set -x


SCRIPT_DIR=$(dirname "$0")
CUPC_DIR="${SCRIPT_DIR}/cupc"

cd $CUPC_DIR

if [ -e "./benchmark_cupc_pcalg_bnlearn.R" ]
then
    echo "Found benchmark script, using existing file. If you don't want this manually copy the file to the pcalg dir"
else
    echo "Copying benchmark_cupc_pcalg_bnlearn.R to pcalg dir"
    cp ../benchmark_cupc_pcalg_bnlearn.R ./benchmark_cupc_pcalg_bnlearn.R
fi

if [ -e "./Skeleton.so" ]
then
    echo "Found Skeleton.so, assuming that all dependencies are installed"
else
    echo "installing dependencies"
    R < "../create-r-libs-user.R" --save
    R < "../install_dependencies.R" --save
    nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu
fi

Rscript benchmark_cupc_pcalg_bnlearn.R -d $1 -c $2 -o "../${3}" -l $4 --libraries $5
# can be used for pinning to one numa node
# numactl --membind 0 --cpunodebind 0


if [ $? != 0 ]; then
  echo "benchmark_cupc_pcalg_bnlearn failed"
  exit 1
else
  exit 0
fi

