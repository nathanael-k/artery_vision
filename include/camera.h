//
// Created by nate on 14.09.20.
//

#ifndef ARTERY_VISION_CAMERA_H
#define ARTERY_VISION_CAMERA_H


#include <string>
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::Vector2d;

class Camera {

public:
    std::string name;

    // in m
    Vector3d origin;

    Vector3d direction;
    Vector3d cameraUp;
    Vector3d cameraLeft;

    // position of pixel 0,0 on the image plane (relative to origin)
    Vector3d topLeftPosition;

    // in m
    double sensorSize = 0;
    double focalLength = 0;

    // pixels
    double resolution = 0;

    // distance every pixel occupies on the sensor
    double pixelDistance;

    Camera() {}

    Camera(std::string metaPath, int index);

    // origin is top left, x is right, y is down (openCV convention)
    // points from origin to the pixel on the image plane
    Vector3d ray(const Vector2d& position) const;

    // returns the image coordinates (at least one of x or y >= 0) on success
    // returns -1 -1 on fail behind camera
    Vector2d projectPoint(const Vector3d& point) const;

    // returns true if the pixel is on the edge of the frustrum
    bool const onEdge(const Eigen::Vector2i& point);

    // returns the image coordinates of the line segment projected onto the imageplane
    // return value is x1, y1, x2, y2
    Eigen::Vector4d const projectLine(const Vector3d &_origin, const Vector3d &_direction);

    // find the closest point between two lines (intersection if the cross), return the distance
    double static intersect(const Vector3d& originA, const Vector3d& directionA,
                     const Vector3d& originB, const Vector3d& directionB, Vector3d& intersection);

    double static intersect(const Camera& camA, const Vector2d& pixelA, const Camera& camB, const Vector2d& pixelB, Vector3d& intersection);
};



#endif //ARTERY_VISION_CAMERA_H
