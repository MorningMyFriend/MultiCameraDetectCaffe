cmake_minimum_required(VERSION 2.8)
project(IntelligentCabinet)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

FIND_PACKAGE(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
        /usr/local/cuda-8.0/include
        ${PROJECT_SOURCE_DIR}/thirdParty/include)
link_directories(${PROJECT_SOURCE_DIR}/thirdParty/lib)

add_subdirectory(src)
add_subdirectory(tests)
