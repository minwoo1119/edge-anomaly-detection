cmake_minimum_required(VERSION 3.16)

project(
    edge_anomaly_detection
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include
)

find_library(
    TENSORRT_LIBRARY
    nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or nvinfer library were not found")
endif()

add_executable(
    edge_anomaly
    src/main.cpp
    src/Preprocessor.cpp
    src/TensorRTInferencer.cpp
    src/MemoryBank.cpp
    src/CpuBruteForceSearch.cpp
    src/PatchCorePostprocessor.cpp
    src/RuntimeConfig.cpp
)

target_include_directories(
    edge_anomaly
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(
    edge_anomaly
    PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    CUDA::cudart
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(edge_anomaly PRIVATE OpenMP::OpenMP_CXX)
endif()
