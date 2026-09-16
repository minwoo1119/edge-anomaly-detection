cmake_minimum_required(VERSION 3.16)

project(
    edge_anomaly_detection
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(
    edge_anomaly
    src/main.cpp
    src/Preprocessor.cpp
)

target_include_directories(
    edge_anomaly
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    edge_anomaly
    PRIVATE
    ${OpenCV_LIBS}
)