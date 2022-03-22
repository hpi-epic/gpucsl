from nvidia/cuda:11.2.0-devel-ubuntu20.04 as gpucsl
SHELL ["/bin/bash", "-c"]

WORKDIR /gpucsl

copy ./gpucsl ./gpucsl/
copy ./benchmarks ./benchmarks/
copy ./scripts ./scripts/
copy ./tests ./tests/

copy ./MANIFEST.in ./
copy ./pyproject.toml ./
copy ./README.md ./
copy ./setup.cfg ./
copy ./setup.py ./


run apt-get update -y && apt-get install -y \
    python3 \
    python3.8-venv \
    python3-pip \
    wget \
    unzip

run python3 -m venv venv && source venv/bin/activate

# install dependencies for GPUCSL
run pip install .
run pip install cupy-cuda112


from gpucsl as gpucsl-benchmarks

run apt-get install -y gfortran \
    libblas-dev \
    liblapack-dev

# install R following: https://cran.r-project.org/bin/linux/ubuntu/
run DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y software-properties-common 
run apt-get install --no-install-recommends -y dirmngr
run wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
run add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
run apt-get install --no-install-recommends -y r-base
run add-apt-repository ppa:c2d4u.team/c2d4u4.0+

run apt-get install -y r-cran-fastica \
    libcurl4-openssl-dev \
    libgmp3-dev

# run the installation here so we do not have to do it while running the benchmarks for the first time 
# and to be sure everything runs without problems (all dependencies are installed)
run ./benchmarks/create-r-libs-user.R
run ./benchmarks/install_dependencies.R

WORKDIR benchmarks/cupc_and_pcalg
run nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu

WORKDIR ../..

# download datasets used for benchmarking
run ./scripts/download-data.sh