FROM osrf/ros:jazzy-desktop

RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

RUN apt-get update -y && apt-get install -y g++-14 gcc-14 cmake build-essential gdb git curl vim libtbb-dev dbus libeigen3-dev sudo iputils-ping unzip wget iproute2 libzstd-dev zstd locales curl

RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc

ENV CC=gcc-14
ENV CXX=g++-14
ENV DOCKER_FLAG_FOR_RUN_SCRIPT=1

# CMAKE_PREFIX_PATH will be updated to include /torch/libtorch
WORKDIR /torch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
RUN unzip *.zip

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
