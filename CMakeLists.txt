cmake_minimum_required(VERSION 3.0)
project(grounding_dino.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
link_directories(${CMAKE_SOURCE_DIR}/thirty_party/onnxruntime/lib)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP_CXX_FOUND Found")
else()
    message(FATAL_ERROR "OpenMP_CXX_FOUND Not Found!")
endif()

FILE(GLOB SRC_LST *.cpp)
add_executable(grounding_dino.cpp ${SRC_LST})
target_link_libraries(grounding_dino.cpp PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
    utf8proc
    OpenMP::OpenMP_CXX
    )

target_include_directories(grounding_dino.cpp PUBLIC 
    ${CMAKE_SOURCE_DIR}/thirty_party/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc
    ${OPENCV_INCLUDE_DIRS}
    )
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirty_party/utf8proc)