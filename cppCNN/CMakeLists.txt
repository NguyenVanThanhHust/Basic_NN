cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 17)

project(linear_sample)
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

add_executable(linear_cpp 
    src/linear.h 
    src/linear.cc 
    src/main.cc
)    

target_link_libraries(linear_cpp 
    ${YAML_CPP_LIBRARIES}
)