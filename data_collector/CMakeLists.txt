cmake_minimum_required(VERSION 3.5)
project(data_collector)
add_compile_options(-std=c++14)

find_package(Threads REQUIRED)

set(MUJOCO_DIR "$ENV{MUJOCO200_HOME}")
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (MUJOCO_DIR)
    message(STATUS "MUJOCO200_HOME at ${MUJOCO_DIR}")
else ()
    message(FATAL_ERROR "Undefined MUJOCO200_HOME env var!")
endif ()

include_directories(${MUJOCO_DIR}/include)

add_executable(data_collector data_collector.cpp)

if (APPLE)
    target_link_libraries(data_collector
            pthread
            ${MUJOCO_DIR}/bin/libmujoco200.dylib
            ${MUJOCO_DIR}/bin/libglfw.3.dylib)
elseif (UNIX AND NOT APPLE)
    target_link_libraries(data_collector
            pthread
            ${MUJOCO_DIR}/bin/libmujoco200.so
            ${MUJOCO_DIR}/bin/libglfw.so.3
            ${MUJOCO_DIR}/bin/libglew.so GL)
else ()
    target_link_libraries(data_collector
            ${MUJOCO_DIR}/bin/mujoco200.dll
            ${MUJOCO_DIR}/bin/glfw3.dll)
endif ()

