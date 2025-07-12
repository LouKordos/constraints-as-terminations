#!/bin/bash
set -exo pipefail

# Default build type
BUILD_TYPE="Release"
BUILD_DIR="/app/build"
LOG_FILE="/app/build.log"
CONTAINER_NAME="sim2real-cat_sim2real-1"
BINARY_NAME="run_policy"
BINARY_ARGV="ens4" # Separated so that it can be passed to gdb as well
# BINARY_NAME="sdk_stand_example ens4"
export CLICOLOR=1
export CLICOLOR_FORCE=1

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build-type=*)
            BUILD_TYPE="${1#*=}"
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1, exiting."
            exit 1
            ;;
    esac
done
        
if [[ -z "${DOCKER_FLAG_FOR_RUN_SCRIPT}" ]]; then
    echo "Host (no docker) detected."
    echo "Installing qemu emulation for arm64 builds..."
    docker run --privileged --rm tonistiigi/binfmt --install all
    echo "Creating buildx builder for multiarch builds if not exists..."
    if ! docker buildx inspect multiarch-builder-${CONTAINER_NAME} >/dev/null 2>&1; then
        echo "Creating multiarch builder…"
        docker buildx create --name multiarch-builder-${CONTAINER_NAME} --driver docker-container --bootstrap
    else
        echo "multiarch builder already exists, skipping creation."
    fi 
    echo "Starting build..."
    COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose --progress plain build --builder multiarch-builder-${CONTAINER_NAME}
    rm -rf ./build || true # Reset build to a clean state, as build cache can cause confusing issues when changing installed deps in the Dockerfile.
    rm -rf ./ros2_ws/{build,install,log} || true # Same for ROS workspace
    docker compose up -d
    echo "Docker container is now running, starting interactive shell. Run /app/build-and-run.sh inside the shell to proceed."
    echo "If you require GUI access, you may need to run (outside the docker container) xhost +local:docker, as well as xhost +SI:localuser:root for ssh X forwarding => Docker => X11"
    echo "You may also choose to docker compose push now for easy deployment. Not automating this due to high bandwidth and time usage."
    echo "Do not forget to export ROS_DOMAIN_ID=0 if you want to communicate with the Go2. Check if it works with ros2 topic list, that should show topics such as /utlidar/cloud."
    docker exec -it $CONTAINER_NAME /bin/bash
else
    echo "Docker detected."
    cd /app
    echo "Building ROS packages..."
    ./bootstrap_ros2_ws.sh
    # source ./ros2_ws/install/setup.bash
    echo "Building main codebase..."
    time cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -S . -B ${BUILD_DIR}
    if [[ ! -f "/tracy-for-capture-built.marker" ]]; then
        echo "-----------------------------------------Setting up automatic tracy profile capture------------------------------------------------"
        cp -r ${BUILD_DIR}/_deps/tracy-src/ /tracy-for-capture/
        rm -r /tracy-for-capture/capture/build || true
        cmake -B /tracy-for-capture/capture/build -S /tracy-for-capture/capture -DCMAKE_BUILD_TYPE=Release -DNO_FILESELECTOR=ON
        cmake --build /tracy-for-capture/capture/build/ --config Release --parallel
        touch "/tracy-for-capture-built.marker"
        echo "-------------------------------------------------Tracy capture setup complete------------------------------------------------------"
    fi
    
    time cmake --build "$BUILD_DIR" -j $(nproc) -v 2>&1 | tee "$LOG_FILE"
    echo "-----------------------------------------------------------------------------------------------------------------------------------"
    # Check for warnings in the build output
    if grep -q "warning:" "$LOG_FILE"; then
        echo -e "\n\033[1;33mWarnings detected during compilation, waiting for confirmation.\033[0m"
        read
    else
        echo -e "\n\033[1;32mNo warnings detected during compilation.\033[0m"
    fi

    pkill -f tracy-capture || echo "No tracy capture instances running."
    mkdir -p /app/logs/tracy-capture/
    mkdir -p /app/tracy-profiles/
    TRACY_LOG_FILE="/app/logs/tracy-capture/tracy-capture-$(date '+%Y-%m-%d-%H-%M-%S').log"
    # Start a new tracy-capture instance and redirect output to the log file 
    /tracy-for-capture/capture/build/tracy-capture -o /app/tracy-profiles/$(date '+%Y-%m-%d-%H-%M-%S').tracy > "$TRACY_LOG_FILE" 2>&1 & 
    TRACY_PID=$!
    echo "Tracy capture started with PID $TRACY_PID. Logs are being written to $TRACY_LOG_FILE"
    chmod -R 777 /app/logs
    chmod -R 777 /app/tracy-profiles

    if [[ "${BUILD_TYPE}" = "Debug" ]]; then
        gdb --args ${BUILD_DIR}/src/$BINARY_NAME $BINARY_ARGV
    else
        ${BUILD_DIR}/src/$BINARY_NAME $BINARY_ARGV
    fi

    echo "Remember to source /app/ros2_ws/install/setup.bash if you are working with ROS custom packages! bashrc already sources /opt/ros/jazzy/setup.bash"
    echo "Also remember to export ROS_DOMAIN_ID=0 if you want to communicate with the Go2."
fi