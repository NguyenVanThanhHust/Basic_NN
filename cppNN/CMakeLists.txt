cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(linear_cpp LANGUAGES CXX CUDA)

message(STATUS "CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)

option(USE_CUDA "Use CUDA" ON)

enable_language("CUDA")

find_package(yaml-cpp REQUIRED)

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

include_directories(
    ${YAML_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} 
    src/tensor.h 
    src/tensor.cc
    src/linear.h 
    src/linear.cc 
    src/main.cc
)    

target_link_libraries(${PROJECT_NAME} 
    ${YAML_CPP_LIBRARIES}
)