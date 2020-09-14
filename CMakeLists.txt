cmake_minimum_required(VERSION 3.17)
project(artery_vision)

set(CMAKE_CXX_STANDARD 14)

enable_testing()
add_subdirectory( test )

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)

add_library(ZHANGSUEN STATIC src/libzhangsuen/zhangsuen.cpp include/zhangsuen.h)
add_library(CAMERA STATIC src/camera/camera.cpp include/camera.h)

target_include_directories(ZHANGSUEN PUBLIC include)
target_compile_features(ZHANGSUEN PUBLIC cxx_std_11)

target_include_directories(CAMERA PUBLIC include)
target_compile_features(CAMERA PUBLIC cxx_std_11)

target_include_directories(test_camera PUBLIC test)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(test)

set(CMAKE_FIND_DEBUG_MODE TRUE)

#find_library(LIBZHANGSUEN zhangsuen lib REQUIRED)
#find_library(LIBCAMERA camera lib REQUIRED)

add_executable(artery_vision src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ZHANGSUEN} ${CAMERA} Eigen3::Eigen)
target_link_libraries(test_camera ${OpenCV_LIBS} ${ZHANGSUEN} ${CAMERA} Eigen3::Eigen)