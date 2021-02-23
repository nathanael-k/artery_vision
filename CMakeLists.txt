cmake_minimum_required(VERSION 3.17)
project(artery_vision)

set(CMAKE_CXX_STANDARD 14)

enable_testing()
add_subdirectory( test )

find_package(OpenCV REQUIRED PATHS /home/nk/Desktop/opencv_build_2 NO_DEFAULT_PATH)
find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(ZHANGSUEN STATIC src/libzhangsuen/zhangsuen.cpp include/zhangsuen.h)
add_library(camera STATIC src/camera/camera.cpp include/camera.h)
add_library(arteryNet STATIC src/arteryNet/arteryNet.cpp include/arteryNet.h)
add_library(imageData STATIC src/imageData/imageData.cpp include/imageData.h)
add_library(imageData2 STATIC src/imageData2/imageData2.cpp include/imageData2.h)
add_library(ball STATIC src/ball/ball.cpp include/ball.h)
add_library(stereo_camera STATIC src/stereo_camera/stereo_camera.cpp include/stereo_camera.h)
add_library(ball_optimizer STATIC src/ball_optimizer/ball_optimizer.cpp include/ball_optimizer.h)


target_include_directories(ZHANGSUEN PUBLIC include)
target_compile_features(ZHANGSUEN PUBLIC cxx_std_11)

target_include_directories(camera PUBLIC include)
target_compile_features(camera PUBLIC cxx_std_11)

target_include_directories(arteryNet PUBLIC include)
target_compile_features(arteryNet PUBLIC cxx_std_11)

target_include_directories(imageData PUBLIC include)
target_compile_features(imageData PUBLIC cxx_std_11)

target_include_directories(imageData2 PUBLIC include)
target_compile_features(imageData2 PUBLIC cxx_std_11)

target_include_directories(stereo_camera PUBLIC include)
target_compile_features(stereo_camera PUBLIC cxx_std_11)

target_include_directories(test_camera PUBLIC test)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(test)

set(CMAKE_FIND_DEBUG_MODE TRUE)

#find_library(LIBZHANGSUEN zhangsuen lib REQUIRED)
#find_library(LIBCAMERA camera lib REQUIRED)

add_executable(artery_vision src/main.cpp)
add_executable(2camtime src/apps/2camtime.cpp src/imageStereoAnalyzer/image_stereo_analyzer.cpp include/image_stereo_analyzer.h)
add_executable(ball_optimization src/apps/ball_optimization.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} imageData camera arteryNet Eigen3::Eigen)
target_link_libraries(test_camera ${OpenCV_LIBS} camera Eigen3::Eigen)
target_link_libraries(test_net arteryNet Eigen3::Eigen)
target_link_libraries(imageData ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(2camtime ${OpenCV_LIBS} imageData camera arteryNet Eigen3::Eigen)
target_link_libraries(ball_optimization ${OpenCV_LIBS} imageData2 camera arteryNet Eigen3::Eigen ball stereo_camera ball_optimizer)