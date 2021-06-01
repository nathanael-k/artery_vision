
add_executable(ball_optimization ball_optimization.cpp)

target_link_libraries(ball_optimization ${OpenCV_LIBS} arteryNet2 artery_replicator)