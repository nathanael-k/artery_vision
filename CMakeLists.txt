cmake_minimum_required(VERSION 3.17)
project(artery_vision)

set(CMAKE_CXX_STANDARD 14)

enable_testing()
add_subdirectory( test )
add_subdirectory( src/apps)

find_package(OpenCV REQUIRED PATHS /home/nk/Desktop/opencv_build_2 NO_DEFAULT_PATH)
find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(ZHANGSUEN STATIC src/libzhangsuen/zhangsuen.cpp include/zhangsuen.h)
add_library(camera STATIC src/camera/camera.cpp include/camera.h)
add_library(arteryNet STATIC src/arteryNet/arteryNet.cpp include/arteryNet.h)
add_library(arteryNet2 STATIC src/arteryNet2/arteryNet2.cpp include/arteryNet2.h)
add_library(imageData STATIC src/imageData/imageData.cpp include/imageData.h)
add_library(imageData2 STATIC src/imageData2/imageData2.cpp include/imageData2.h)
add_library(ball STATIC src/ball/ball.cpp include/ball.h)
add_library(stereo_camera STATIC src/stereo_camera/stereo_camera.cpp include/stereo_camera.h)
add_library(ball_optimizer STATIC src/ball_optimizer/ball_optimizer.cpp include/ball_optimizer.h)
add_library(artery_replicator STATIC src/artery_replicator/artery_replicator.cpp include/artery_replicator.h)



include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(test)

set(CMAKE_FIND_DEBUG_MODE TRUE)

#find_library(LIBZHANGSUEN zhangsuen lib REQUIRED)
#find_library(LIBCAMERA camera lib REQUIRED)



target_link_libraries(test_camera ${OpenCV_LIBS} camera Eigen3::Eigen)
target_link_libraries(test_net arteryNet Eigen3::Eigen)
target_link_libraries(imageData ${OpenCV_LIBS} Eigen3::Eigen)

target_link_libraries(imageData2 ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(stereo_camera camera)
target_link_libraries(ball_optimizer ${OpenCV_LIBS} ball imageData2)
target_link_libraries(artery_replicator ${OpenCV_LIBS} arteryNet2 stereo_camera ball_optimizer)