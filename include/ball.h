#pragma once

#include <Eigen/Core>

class Ball {
    Eigen::Vector3d center;
    double radius;
    double confidence;
};