cmake_minimum_required(VERSION 3.16)

project(
    edge_anomaly_detection
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include
)

include(CTest)

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
    src/Benchmark.cpp
    src/NpyWriter.cpp
    src/CudaBruteForceSearch.cu
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
    target_compile_definitions(edge_anomaly PRIVATE EDGE_HAS_OPENMP=1)
endif()

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    OUTPUT_VARIABLE EDGE_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT EDGE_GIT_COMMIT)
    set(EDGE_GIT_COMMIT unknown)
endif()
execute_process(
    COMMAND git status --porcelain
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    OUTPUT_VARIABLE EDGE_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(EDGE_GIT_STATUS)
    set(EDGE_GIT_DIRTY 1)
else()
    set(EDGE_GIT_DIRTY 0)
endif()
target_compile_definitions(
    edge_anomaly
    PRIVATE
    EDGE_GIT_COMMIT="${EDGE_GIT_COMMIT}"
    EDGE_GIT_DIRTY=${EDGE_GIT_DIRTY}
    EDGE_BUILD_TYPE="$<CONFIG>"
)

if(BUILD_TESTING)
    add_executable(
        edge_runtime_tests
        tests/runtime_tests.cpp
        src/MemoryBank.cpp
        src/CpuBruteForceSearch.cpp
        src/PatchCorePostprocessor.cpp
        src/NpyWriter.cpp
        src/Preprocessor.cpp
        src/CudaBruteForceSearch.cu
    )
    target_include_directories(
        edge_runtime_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(edge_runtime_tests PRIVATE ${OpenCV_LIBS} CUDA::cudart)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(edge_runtime_tests PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(edge_runtime_tests PRIVATE EDGE_HAS_OPENMP=1)
    endif()
    add_test(NAME edge_runtime_tests COMMAND edge_runtime_tests)
endif()
