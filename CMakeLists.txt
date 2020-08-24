cmake_minimum_required(VERSION 3.17)
project(artery_vision)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(artery_vision src/main.cpp)

add_library(zhangsuen STATIC src/libzhangsuen/zhangsuen.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(include)
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_library(LIBZHANGSUEN zhangsuen lib REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}  ${LIBZHANGSUEN})