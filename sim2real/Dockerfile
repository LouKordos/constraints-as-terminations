ARG ROS_DISTRO=jazzy

FROM ros:$ROS_DISTRO-ros-core
ENV ROS_DISTRO=$ROS_DISTRO
RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4
# Essential stuff
RUN apt-get update -y && apt-get install -y g++-14 gcc-14 cmake build-essential gdb git curl vim fzf ripgrep libtbb-dev dbus git-lfs libeigen3-dev libboost-python-dev libboost-numpy-dev libpcl-dev python3-dev python3-numpy python3-matplotlib libgoogle-glog-dev tmux sudo iputils-ping unzip wget iproute2 libzstd-dev zstd locales curl dbus dbus-x11 x11-apps xauth tree net-tools cyclonedds-tools netcat-openbsd libboost-all-dev libyaml-cpp-dev

# Need to build from source due to arm64 on Go2 Jetson
RUN apt-get update -y && apt-get install -y build-essential cmake git python3 python3-pip python3-venv libopenblas-dev libomp-dev libopenmpi-dev ninja-build libyaml-dev libjpeg-dev libpng-dev
RUN apt-get upgrade -y
WORKDIR /torch
RUN git clone --recursive https://github.com/pytorch/pytorch.git
WORKDIR /torch/pytorch
RUN git checkout e2d141dbde55c2a4370fac5165b0561b6af4798b && git submodule sync && git submodule update --init --recursive
WORKDIR /torch/pytorch/build
RUN apt-get install -y python3-typing-extensions
RUN CFLAGS="-Wno-error" CXXFLAGS="-Wno-error" cmake -G Ninja --compile-no-warning-as-error \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DUSE_CUDA=OFF \
      -DUSE_ROCM=OFF \
      -DUSE_CUDNN=OFF \
      -DUSE_NCCL=OFF \
      -DUSE_TENSORPIPE=OFF \
      -DUSE_FBGEMM=OFF \
      -DUSE_XNNPACK=OFF \
      -DUSE_PYTORCH_QNNPACK=OFF \
      -DUSE_NNPACK=OFF \
      -DUSE_DISTRIBUTED=OFF \
      -DUSE_KINETO=OFF \
      -DBUILD_TEST=OFF \
      -DUSE_OPENCL=OFF \
      -DUSE_SYSTEM_ONNX=OFF \
      -DUSE_NNAPI=OFF \
      -DBUILD_PYTHON=OFF \
      -DCMAKE_C_FLAGS="-Wno-error" \
      -DCMAKE_CXX_FLAGS="-Wno-error" \
      -DCMAKE_CUDA_FLAGS="-Wno-error" \
      -DCMAKE_INSTALL_PREFIX=/torch/libtorch \
      ..
RUN CFLAGS="-Wno-error" CXXFLAGS="-Wno-error" cmake --build . --target install

# ROS2 stuff
RUN apt-get update -y && apt-get install -y ros-dev-tools ros-$ROS_DISTRO-desktop ros-$ROS_DISTRO-rmw-cyclonedds-cpp ros-$ROS_DISTRO-rosidl-generator-dds-idl ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-pointcloud-to-laserscan

# Set this AFTER libtorch due to compiler bug in gcc14!
ENV CC=gcc-14
ENV CXX=g++-14

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

WORKDIR /livox-sdk
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git
WORKDIR /livox-sdk/Livox-SDK2/build
# Patches needed for C++23 build
RUN sed -i '/^#include /a #include <cstdint>' ../sdk_core/logger_handler/file_manager.h
RUN sed -i '/^#include /a #include <cstdint>' ../sdk_core/comm/define.h
RUN cmake .. && make -j
RUN make install

# go2_odometry
WORKDIR /inekf
RUN git clone https://github.com/inria-paris-robotics-lab/invariant-ekf.git
WORKDIR /inekf/invariant-ekf/build
RUN git checkout 6047ef98b8984ad041ce10849b351df17de4dfe8
# Patch minimum required CMakeLists.txt
RUN sed -E -i.bak 's#^(cmake_minimum_required\(\s*VERSION\s*)[0-9]+(\.[0-9]+)*#\1 3.20#' ../CMakeLists.txt
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=ON -DPYTHON_DEB_LAYOUT=ON ..
RUN make -j && make install

# RUN AFTER ALL DEPENDENCY INSTALLATIONS!
RUN ldconfig
ENV DOCKER_FLAG_FOR_RUN_SCRIPT=1

RUN rosdep init
RUN rosdep update
RUN echo '' >> /root/.bashrc
RUN echo 'eval "$(fzf --bash)"' >> /root/.bashrc
# RUN echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc
RUN echo '# Remind user to source custom ROS2 workspace' >> /root/.bashrc
RUN echo 'echo "Remember to source /app/ros2_ws/install/setup.bash OR /app/odom_alternative_ws/install/setup.bash if you are working with ROS!"' >> /root/.bashrc
RUN echo 'echo "Also remember to export ROS_DOMAIN_ID=0 if you want to communicate with the Go2."' >> /root/.bashrc
ENV ROS_DOMAIN_ID=69
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY . /app
COPY .vimrc /root/.vimrc
WORKDIR /app

# Using CMD instead of ENTRYPOINT here so that docker-compose.yml for vs code can override it.
CMD sleep infinity
