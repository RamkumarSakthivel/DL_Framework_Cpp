cmake_minimum_required(VERSION 3.10)

project(DL_Framework_Cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create the main library
add_library(DL_Framework_Cpp ${SOURCES})

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add executable (if needed later)
add_executable(main src/main.cpp
                    src/tensor.cpp)
target_link_libraries(main DL_Framework_Cpp)
