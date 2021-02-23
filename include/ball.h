#pragma once

#include <Eigen/Core>
#include <bits/stdint-intn.h>
#include <sys/types.h>
#include <vector>

struct Ball {
  Eigen::Vector3d direction;
  Eigen::Vector3d center_m;
  double radius_m;
  double confidence;

  int connections_A = -1;
  int connections_B = -1;

  Ball next_ball();

  void project_to_surface(const Ball &other, double radius_factor);
};

const Ball &find_ball_at(const std::vector<Ball> &balls,
                         const Ball &query_ball);

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
  Circle(Eigen::Vector2d location_px, double radius_px, double angle_deg);

  Circle(Eigen::Vector2d location_px, double radius_px,
         Eigen::Vector2d point_at_px);

  double angle_deg() const;

  double angle_rad() const;

  void set_angle_deg(double new_angle_deg);

  void set_angle_rad(double new_angle_rad);

  Eigen::Vector2d direction_px() const;

  void point_at(Eigen::Vector2d point_at_px);

  void apply_gradient(const CircleGradient &gradient, double dx);
};
