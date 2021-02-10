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

    BallOptimizer(Ball& ball, const imageData& image_data, const StereoCamera& stereo_camera);

    // optimize the properties of ball in reference to the image data captured through stereo cameras
    void optimize(const uint16_t steps, const uint8_t frame_index);





private:
Ball& ball;
const imageData& image_data;
const StereoCamera& stereo_camera;
const 

    // the actual optimization step
    void step(const double dx, const uint8_t frame_index);

    // generates a circle that represent the projection of a ball onto the image of the camera
    Circle project_ball(uint8_t camera_index) const;

    // generates the gradient of the ball in camera coordinates
    CircleGradient get_gradient(const uint8_t camera_index, const uint8_t frame_index) const;


};