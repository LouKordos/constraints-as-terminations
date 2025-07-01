#!/bin/bash
set -exo pipefail

# Default build type
BUILD_TYPE="Release"
BUILD_DIR="/app/build"
LOG_FILE="/app/build.log"
CONTAINER_NAME="sim2real-cat_sim2real-1"
# BINARY_NAME="run_policy ens4" # Adjust ethernet interface as needed
BINARY_NAME="sdk_stand_example ens4"
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
    docker compose --progress plain up -d
    echo "Docker container is now running, starting interactive shell. Run /app/build-and-run.sh inside the shell to proceed."
    docker exec -it $CONTAINER_NAME /bin/bash
else
    echo "Docker detected."
    cd /app
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
        gdb ${BUILD_DIR}/src/$BINARY_NAME
    else
        ${BUILD_DIR}/src/$BINARY_NAME
    fi
fi