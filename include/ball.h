#pragma once

#include <Eigen/Core>

class Ball {
    Eigen::Vector3d direction;
    Eigen::Vector3d center_cm;
    double radius_cm;
    double confidence;
};