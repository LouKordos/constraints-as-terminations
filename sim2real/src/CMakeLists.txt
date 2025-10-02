set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -fverbose-asm -Wall -Wextra -march=native -mtune=native -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -ffold-simple-inlines")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

include(FetchContent)
FetchContent_Declare (
  tracy
  GIT_REPOSITORY https://github.com/wolfpld/tracy.git
  GIT_TAG 7f6efe3bd4a06d6e460465c6f0f47f32df25685d
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(tracy)
option(TRACY_ENABLE "" ON)

FetchContent_Declare (
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG 15f539685bbb21b3bb01475ac89c6d8d53b5b60f 
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(spdlog)

find_package(Threads REQUIRED)
find_package(unitree_sdk2 REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/torch/libtorch/")
find_package(Torch REQUIRED)
find_package(cppzmq REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(run_policy run_policy.cpp)
target_include_directories(run_policy PUBLIC ${sensor_msgs_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS} ${geometry_msgs_INCLUDE_DIRS})
target_link_libraries(run_policy PUBLIC spdlog::spdlog TracyClient atomic unitree_sdk2 atomic Threads::Threads "${TORCH_LIBRARIES}" cppzmq)

add_executable(read_observations read_observations.cpp)
target_link_libraries(read_observations PUBLIC spdlog::spdlog TracyClient atomic unitree_sdk2 Threads::Threads)

add_executable(test_pytorch_policy test_pytorch_policy.cpp)
target_link_libraries(test_pytorch_policy PUBLIC spdlog::spdlog TracyClient atomic unitree_sdk2 Threads::Threads "${TORCH_LIBRARIES}" cppzmq)

add_executable(sdk_stand_example sdk_stand_example.cpp)
target_link_libraries(sdk_stand_example PUBLIC unitree_sdk2)