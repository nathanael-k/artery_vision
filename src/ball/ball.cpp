#include <ball.h>

#include <Eigen/Core>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <limits>
#include <sys/types.h>
#include <vector>
#include <list>

Ball Ball::next_ball() {
  return {direction, center_m - direction * radius_m * 1.5, radius_m, 0};
}

void Ball::project_to_surface(const Ball &other, double radius_factor) {
  Eigen::Vector3d delta =
      (center_m - other.center_m).normalized() * other.radius_m * radius_factor;
  center_m = other.center_m + delta;
}

std::list<Ball*>::iterator find_ball_at(std::list<Ball*> &balls,
                         const Eigen::Vector3d location_m) {
  auto it = balls.begin();
  for (it; it != balls.end(); it++) {
    double distance = (location_m - (*it)->center_m).norm();
    if (distance < (*it)->radius_m)
      break;
  }
  return it;
}

Circle::Circle(Eigen::Vector2d location_px, double radius_px, double angle_deg)
    : location_px(location_px), radius_px(radius_px) {
  set_angle_deg(angle_deg);
}

Circle::Circle(Eigen::Vector2d location_px, double radius_px,
               Eigen::Vector2d point_at_px)
    : location_px(location_px), radius_px(radius_px) {
  point_at(point_at_px);
}

double Circle::angle_deg() const { return angle_deg_; }

double Circle::angle_rad() const { return angle_deg_ * M_PI / 180.; }

void Circle::set_angle_deg(double new_angle_deg) {
  if (new_angle_deg >= 360)
    angle_deg_ = new_angle_deg;
  else if (new_angle_deg < 0)
    angle_deg_ = new_angle_deg + 360;
  else
    angle_deg_ = new_angle_deg;
}

void Circle::set_angle_rad(double new_angle_rad) {
  double new_angle_deg = new_angle_rad * 180 * M_1_PI;
  set_angle_deg(new_angle_deg);
}

Eigen::Vector2d Circle::direction_px() const {
  Eigen::Vector2d direction_px = location_px;
  direction_px.x() += sin(angle_rad()) * radius_px;
  direction_px.y() -= cos(angle_rad()) * radius_px;
  return direction_px;
}

void Circle::point_at(Eigen::Vector2d point_at_px) {
  Eigen::Vector2d direction = point_at_px - location_px;
  set_angle_rad(atan2(direction.x(), -direction.y()));
}

void Circle::apply_gradient(const CircleGradient &gradient, double dx) {
  // top priority is the rotation, because the other two gradients can make no
  // sense if direction is not aligned. if the rotation gradients are low
  // enough, we focus on the scale if both rotation and scale match reasonably
  // well, we optimize the translation perpendicular
  double rotation_factor = 1.0;
  double scale_factor = 0.01;
  double translation_factor = 0.1;

  double rotation_grad = gradient.rotation_grad_1 + gradient.rotation_grad_2;
  double scale_grad = gradient.scale_grad_1 + gradient.scale_grad_2;
  double translation_grad =
      gradient.translation_grad_1 + gradient.translation_grad_2;

  angle_deg_ += dx * rotation_factor * rotation_grad;
  if (abs(rotation_grad) < 0.1) {
    radius_px *= 1.0 + dx * scale_factor * scale_grad;
    location_px.x() +=
        dx * translation_factor * translation_grad * cos(angle_rad());
    location_px.y() +=
        dx * translation_factor * translation_grad * sin(angle_rad());
  }
}

Circle mid_circle (const std::vector<Circle>& circles) {
  Eigen::Vector2d center_px(0, 0);
  double radius_px(0);

  for (const auto& circle : circles) {
    center_px += circle.location_px;
    radius_px += circle.radius_px;
  }

  center_px /= circles.size();
  radius_px /= circles.size();

  return {center_px, radius_px, circles[0].location_px};
}


std::vector<double> relative_angles(const std::vector<Circle>& circles, const Circle& query) {
  std::vector<double> ret;
  for (const auto& circle : circles) {
    Eigen::Vector2d direction = circle.location_px - query.location_px;
    ret.emplace_back(atan2(direction.y(), direction.x()));
  }
  return ret;
}

size_t max_angle_difference(const std::vector<double>& less, const std::vector<double>& more) {
  size_t max_index;
  double max_angle = 0;
  for (size_t i = 0; i < less.size(); i++) {
    size_t min_index;
    double min_angle = std::numeric_limits<double>::max();
    for (size_t j = 0; j < more.size(); j++) {
      double angle = std::fmod(std::abs(less[i] - more[j]),  180.);
      if (angle < min_angle) {
        min_angle = angle;
        min_index = j;
      }
    }
    if (min_angle > max_angle) {
      max_angle = min_angle;
      max_index = i;
    }
  }
  return max_index;
}

