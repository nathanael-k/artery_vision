#pragma once

#include <ball.h>
#include <stereo_camera.h>
#include <imageData2.h>

// a circle is the projection of a ball onto the image plane.
// note that real 3d projection would result in skewed circles, but we assume
// long focal length, so a circle is a valid approximation
struct Circle; 

// the quality and the direction we should optimize the circle to
struct CircleGradient;

class BallOptimizer {
public:

    BallOptimizer(Ball& ball, const StereoCamera& stereo_camera, const Circle& circle_A, const Circle& circle_B);

    // optimize the properties of ball in reference to the image data captured through stereo cameras
    void optimize(const uint16_t steps, const uint8_t frame_index);

// private:
public:
Ball& ball;
const StereoCamera& stereo_camera;

    // the actual optimization step
    void step(const double dx, const uint8_t frame_index);

    // generates a circle that represent the projection of a ball onto the image of the camera
    Circle project_circle(uint8_t camera_index) const;

    // based on a circle for each camera, estimate the best fit of a ball in 3d space
    void triangulate_ball(const Circle& circle_A, const Circle& circle_B);

    // project the ball to the two circles
    void project_circles(Circle& out_circle_A, Circle& out_circle_B);
        
    // generates the gradient of the ball in camera coordinates
    CircleGradient get_gradient(const Circle& circle, const uint8_t camera_index, const uint8_t frame_index) const;
 
    

};