#pragma once

#include <Eigen/Core>
#include <bits/stdint-intn.h>
#include <sys/types.h>

struct Ball {
    Eigen::Vector3d direction;
    Eigen::Vector3d center_m;
    double radius_m;
    double confidence;
};

struct CircleGradient {
    double quality;
    double rotation_grad_1, rotation_grad_2;
    double scale_grad_1, scale_grad_2;
    double translation_grad_1, translation_grad_2;
};

struct Circle {
    Eigen::Vector2d location_px;
    double radius_px;
    int16_t connections = -1;

private:
    double angle_deg_;

public:

Circle(Eigen::Vector2d location_px, double radius_px, double angle_deg) : 
    location_px(location_px), radius_px(radius_px) 
{
    set_angle_deg(angle_deg);
}

Circle(Eigen::Vector2d location_px, double radius_px, Eigen::Vector2d point_at_px) : 
    location_px(location_px), radius_px(radius_px) 
{
    point_at(point_at_px);
}

    double angle_deg() const {
        return angle_deg_;
    }

    double angle_rad() const {
        return angle_deg_ * M_PI / 180.;
    }

    void set_angle_deg(double new_angle_deg) {
        if (new_angle_deg > 90)
            angle_deg_ = new_angle_deg - 180;
        else if (new_angle_deg <= -90)
            angle_deg_ = new_angle_deg + 180;
        else
            angle_deg_ = new_angle_deg;
    }

    void set_angle_rad(double new_angle_rad) {
        double new_angle_deg = new_angle_rad * 180 * M_1_PI;
        if (new_angle_deg > 90)
            angle_deg_ = new_angle_deg - 180;
        else if (new_angle_deg <= -90)
            angle_deg_ = new_angle_deg + 180;
        else
            angle_deg_ = new_angle_deg;
    }

    Eigen::Vector2d direction_px() const {
        Eigen::Vector2d direction_px = location_px;
        direction_px.x() += sin(angle_rad()) * radius_px;
        direction_px.y() -= cos(angle_rad()) * radius_px;
        return direction_px;
    }

    void point_at(Eigen::Vector2d point_at_px) {
        Eigen::Vector2d direction = point_at_px - location_px;
        set_angle_rad(atan2(direction.x(), -direction.y()));
    }

    void apply_gradient(const CircleGradient& gradient, double dx) {
        // top priority is the rotation, because the other two gradients can make no sense if direction is not aligned.
        // if the rotation gradients are low enough, we focus on the scale
        // if both rotation and scale match reasonably well, we optimize the translation perpendicular
        double rotation_factor = 1.0;
        double scale_factor = 0.01;
        double translation_factor = 0.1;

        double rotation_grad = gradient.rotation_grad_1 + gradient.rotation_grad_2;
        double scale_grad = gradient.scale_grad_1 + gradient.scale_grad_2;
        double translation_grad = gradient.translation_grad_1 + gradient.translation_grad_2;

        angle_deg_ += dx * rotation_factor * rotation_grad;        
        if (abs(rotation_grad) < 0.1) {
            radius_px *= 1.0 + dx * scale_factor * scale_grad;
            location_px.x() += dx * translation_factor * translation_grad * cos(angle_rad());
            location_px.y() += dx * translation_factor * translation_grad * sin(angle_rad());
        }
    }
};

