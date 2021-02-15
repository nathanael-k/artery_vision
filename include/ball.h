#pragma once

#include <Eigen/Core>

struct Ball {
    Eigen::Vector3d direction;
    Eigen::Vector3d center_m;
    double radius_m;
    double confidence;
};

struct Circle {
    Eigen::Vector2d location_px;
    double radius_px;

private:
    double angle_deg_;

public:

Circle(Eigen::Vector2d location_px, double radius_px, double angle_deg) : 
    location_px(location_px), radius_px(radius_px) 
{
    set_angle(angle_deg);
}

    double angle_deg() const {
        return angle_deg_;
    }

    double angle_rad() const {
        return angle_deg_ * M_PI / 180.;
    }

    void set_angle(double new_angle_deg) {
        if (new_angle_deg > 90)
            angle_deg_ = new_angle_deg - 180;
        else if (new_angle_deg <= -90)
            angle_deg_ = new_angle_deg + 180;
        else
            angle_deg_ = new_angle_deg;
    }
};

struct CircleGradient {
    double quality;
    double rotation_grad_1, rotation_grad_2;
    double scale_grad_1, scale_grad_2;
    double translation_grad_1, translation_grad_2;
};