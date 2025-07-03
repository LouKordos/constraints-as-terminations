FROM osrf/ros:jazzy-desktop

RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

RUN apt-get update -y && apt-get install -y g++-14 gcc-14 cmake build-essential gdb git curl vim libtbb-dev dbus libeigen3-dev sudo iputils-ping unzip wget iproute2 libzstd-dev zstd locales curl

RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

ENV CC=gcc-14
ENV CXX=g++-14
ENV DOCKER_FLAG_FOR_RUN_SCRIPT=1

# CMAKE_PREFIX_PATH will be updated to include /torch/libtorch
#WORKDIR /torch
#RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
#RUN unzip *.zip

# Need to build from source due to arm64 on Go2 Jetson
RUN apt update -y && apt install -y build-essential cmake git python3 python3-pip python3-venv libopenblas-dev libomp-dev libopenmpi-dev ninja-build libyaml-dev libjpeg-dev libpng-dev
WORKDIR /torch
RUN git clone --recursive https://github.com/pytorch/pytorch.git
WORKDIR /torch/pytorch
RUN git checkout e2d141dbde55c2a4370fac5165b0561b6af4798b && git submodule sync && git submodule update --init --recursive
WORKDIR /torch/pytorch/build
RUN apt install -y python3-typing-extensions
RUN CFLAGS="-Wno-error" CXXFLAGS="-Wno-error" cmake -G Ninja --compile-no-warning-as-error \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DUSE_CUDA=OFF \
      -DUSE_ROCM=OFF \
      -DUSE_CUDNN=OFF \
      -DUSE_MIOPEN=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_TENSORPIPE=OFF \
      -DUSE_FBGEMM=OFF \
      -DUSE_QNNPACK=OFF \
      -DUSE_XNNPACK=OFF \
      -DUSE_NNPACK=OFF \
      -DBUILD_CAFFE2=OFF \
      -DBUILD_CAFFE2_OPS=OFF \
      -DUSE_DISTRIBUTED=OFF \
      -DUSE_KINETO=OFF \
      -DBUILD_TEST=OFF \
      -DUSE_OPENCL=OFF \
      -DUSE_OPENCV=OFF \
      -DUSE_SYSTEM_ONNX=OFF \
      -DUSE_NNAPI=OFF \
      -DBUILD_PYTHON=OFF \
      -DTREAT_WARNINGS_AS_ERRORS=OFF \
      -DCMAKE_C_FLAGS="-Wno-error" \
      -DCMAKE_CXX_FLAGS="-Wno-error" \
      -DCMAKE_CUDA_FLAGS="-Wno-error" \
      -DCMAKE_INSTALL_PREFIX=/torch/libtorch \
      ..
RUN CFLAGS="-Wno-error" CXXFLAGS="-Wno-error" cmake --build . --target install

WORKDIR /go2_sdk
RUN git clone https://github.com/unitreerobotics/unitree_sdk2.git
WORKDIR /go2_sdk/unitree_sdk2/build
RUN git checkout 66cc92816dd4af720e35b75f7dbd5be35381b68b
RUN cmake ..
RUN make -j
RUN make install

WORKDIR /libzmq
RUN git clone https://github.com/zeromq/libzmq.git
WORKDIR /libzmq/libzmq
RUN git checkout 622fc6dde99ee172ebaa9c8628d85a7a1995a21d
WORKDIR /libzmq/libzmq/build
RUN cmake ..
RUN make -j install

WORKDIR /cppzmq
RUN git clone https://github.com/zeromq/cppzmq.git
WORKDIR /cppzmq/cppzmq/build
RUN git checkout 3bcbd9dad2f57180aacd4b4aea292a74f0de7ef4
RUN cmake -DCPPZMQ_BUILD_TESTS=OFF ..
RUN make -j install

# RUN AFTER ALL DEPENDENCY INSTALLATIONS!
RUN ldconfig

COPY . /app
WORKDIR /app

# Using CMD instead of ENTRYPOINT here so that docker-compose.yml for vs code can override it.
CMD sleep infinity
