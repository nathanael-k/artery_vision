#include "arteryNet.h"
#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "stereo_camera.h"

#include <ball_optimizer.h>

BallOptimizer::BallOptimizer(Ball &ball, const StereoCamera &stereo_camera,
                             const Circle &circle_A, const Circle &circle_B)
    : ball(ball), stereo_camera(stereo_camera) {
  ball = triangulate_ball(circle_A, circle_B, stereo_camera);
};

BallOptimizer::BallOptimizer(Ball &ball, const StereoCamera &stereo_camera)
    : ball(ball), stereo_camera(stereo_camera){};

void BallOptimizer::optimize(const uint16_t steps, const uint8_t frame_index) {
  for (int i = 0; i < steps; i++) {
    step(1.0, frame_index);
  }
};

void BallOptimizer::optimize_constrained(const uint16_t steps,
                                         const uint8_t frame_index,
                                         const Ball &constrain,
                                         double radius_factor) {
                                           for (int i = 0; i < steps; i++) {
    step_constrained(1.0, frame_index, constrain, radius_factor);
  }
                                         }

void BallOptimizer::step(const double dx, const uint8_t frame_index) {
  // calculate CircleGradients for both cameras

  Circle circleA = project_circle(0);
  Circle circleB = project_circle(1);

  CircleGradient gradA = get_gradient(circleA, 0, frame_index);
  // cv::waitKey(0);
  CircleGradient gradB = get_gradient(circleB, 1, frame_index);
  // cv::waitKey(0);

  std::cout << gradA.quality << "    " << gradB.quality << "\n";

  circleA.apply_gradient(gradA, 1);
  circleB.apply_gradient(gradB, 1);

  ball = triangulate_ball(circleA, circleB, stereo_camera);
}

void BallOptimizer::step_constrained(const double dx, const uint8_t frame_index,
                                     const Ball &constrain,
                                     double radius_factor) {
  step(dx, frame_index);
  ball.project_to_surface(constrain, radius_factor);
}

Circle BallOptimizer::project_circle(uint8_t camera_index) const {
  assert(camera_index == 0 || camera_index == 1);

  const Camera &cam =
      (camera_index) ? stereo_camera.camera_B : stereo_camera.camera_A;

  Circle ret = {cam.projectPoint(ball.center_m),
                cam.estimate_radius_image_px(ball.center_m, ball.radius_m), 0};

  Eigen::Vector2d A_direction_px =
      cam.projectPoint(ball.center_m + ball.direction) - ret.location_px;
  ret.set_angle_rad(atan2(A_direction_px.x(), -A_direction_px.y()));
  return ret;
}

CircleGradient BallOptimizer::get_gradient(const Circle &circle,
                                           uint8_t camera_index,
                                           const uint8_t frame_index) const {

  // setting:
  uint8_t derivative_kernel_size = 31;

  cv::Point location = {(int)circle.location_px.x(),
                        (int)circle.location_px.y()};
  double angle_deg = circle.angle_deg();
  double angle_rad = angle_deg / 180. * M_PI;
  // estimate radius from distance transform...
  const double &radius = circle.radius_px;

  // source rectangle
  cv::Mat patch;

  double sinacosa = abs(sin(angle_rad)) + abs(cos(angle_rad));
  uint32_t edge_2 = ceil((6 * radius * sinacosa) / 2);

  // if necessary, take larger source patch
  if (edge_2 < 16)
    edge_2 = 16;

  cv::Mat src_patch = grab_region(
      location, edge_2,
      stereo_camera.image_data(camera_index).threshold[frame_index]);

  src_patch.convertTo(src_patch, CV_32F, 1. / 255.);

  // rotate
  double scale = derivative_kernel_size / (6 * radius);
  cv::Mat derot;
  cv::Point center = cv::Point(edge_2, edge_2);
  auto rot = cv::getRotationMatrix2D(center, angle_deg, scale);
  cv::warpAffine(src_patch, derot, rot, cv::Size(2 * edge_2, 2 * edge_2));

  // only use center
  float border = (derot.cols - derivative_kernel_size) / 2;
  auto rect =
      cv::Rect(border, border, derivative_kernel_size, derivative_kernel_size);
  patch = derot(rect);

  cv::imshow("Kernel", patch);

  // inverse
  cv::Mat patch_inv = 1 - patch;

  // gaussian weighting
  cv::Mat gauss_kernel =
      100 * (cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F) *
             cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F).t());

  patch = gauss_kernel.mul(patch);
  patch_inv = gauss_kernel.mul(patch_inv);

  // calculate areas of interest
  cv::Rect A1, A2, A3, A4, B1, B2, B3, B4;
  A1 = cv::Rect(0, 0, 10, 15);
  A2 = cv::Rect(10, 0, 5, 15);
  A3 = cv::Rect(15, 0, 6, 15);
  A4 = cv::Rect(21, 0, 10, 15);
  B1 = cv::Rect(0, 15, 10, 16);
  B2 = cv::Rect(10, 15, 5, 16);
  B3 = cv::Rect(15, 15, 6, 16);
  B4 = cv::Rect(21, 15, 10, 16);

  Eigen::MatrixXd area(2, 4);
  area << cv::sum(patch(A1))[0], cv::sum(patch(A2))[0], cv::sum(patch(A3))[0],
      cv::sum(patch(A4))[0], cv::sum(patch(B1))[0], cv::sum(patch(B2))[0],
      cv::sum(patch(B3))[0], cv::sum(patch(B4))[0];

  Eigen::MatrixXd area_inv(2, 4);
  area_inv << cv::sum(patch_inv(A1))[0], cv::sum(patch_inv(A2))[0],
      cv::sum(patch_inv(A3))[0], cv::sum(patch_inv(A4))[0],
      cv::sum(patch_inv(B1))[0], cv::sum(patch_inv(B2))[0],
      cv::sum(patch_inv(B3))[0], cv::sum(patch_inv(B4))[0];

  // Selection matrices
  Eigen::MatrixXd background(2, 4);
  background << 1, 0, 0, 1, 1, 0, 0, 1;
  Eigen::MatrixXd artery(2, 4);
  artery << 0, 1, 1, 0, 0, 1, 1, 0;

  // Rotation selection matrices
  Eigen::MatrixXd inner_rot(2, 4);
  inner_rot << 0, 1, -1, 0, 0, -1, 1, 0;
  Eigen::MatrixXd outer_rot(2, 4);
  outer_rot << -1, 0, 0, 1, 1, 0, 0, -1;

  // Scale selection matrices
  Eigen::MatrixXd scale_1(2, 4);
  scale_1 << 0, 0, 0, 0, 0, -1, -1, 0;
  Eigen::MatrixXd scale_2(2, 4);
  scale_2 << 0, 0, 0, 0, 1, 0, 0, 1;

  // Translation selection matrices
  Eigen::MatrixXd translation_1(2, 4);
  translation_1 << 0, 1, -1, 0, 0, 1, -1, 0;
  Eigen::MatrixXd translation_2(2, 4);
  translation_2 << -1, 0, 0, 1, -1, 0, 0, 1;

  // match quality is white pixels where the artery should be minus white pixels
  // where the background should be
  double quality =
      (area.cwiseProduct(artery).sum() - area.cwiseProduct(background).sum()) /
      cv::sum(gauss_kernel)[0];

  // calculate rotation gradient
  double inner_rot_grad = inner_rot.cwiseProduct(area_inv).sum();
  double outer_rot_grad = outer_rot.cwiseProduct(area).sum();

  // calculate scale gradient
  double scale_1_grad = scale_1.cwiseProduct(area_inv).sum();
  double scale_2_grad = scale_2.cwiseProduct(area).sum();

  // calculate translation gradient
  double translation_1_grad = translation_1.cwiseProduct(area_inv).sum();
  double translation_2_grad = translation_2.cwiseProduct(area).sum();

  // return the gradient
  return {
      quality,      inner_rot_grad,     outer_rot_grad,     scale_1_grad,
      scale_2_grad, translation_1_grad, translation_2_grad,
  };
}

cv::Mat grab_region(cv::Point center, int radius, const cv::Mat &source) {
  int min_x, min_y, max_x, max_y;
  min_x = center.x - radius;
  min_y = center.y - radius;
  max_x = center.x + radius;
  max_y = center.y + radius;

  auto dim = source.size();

  int g_min_x, g_min_y, g_max_x, g_max_y;
  g_min_x = std::max(0, min_x);
  g_min_y = std::max(0, min_y);
  g_max_x = std::min(max_x, dim.width - 1);
  g_max_y = std::min(max_y, dim.height - 1);

  cv::Mat ret;
  cv::Rect grab_area =
      cv::Rect(cv::Point(g_min_x, g_min_y), cv::Point(g_max_x, g_max_y));
  cv::Mat buff = source(grab_area);
  cv::copyMakeBorder(buff, ret, g_min_y - min_y, max_y - g_max_y,
                     g_min_x - min_x, max_x - g_max_x, cv::BORDER_CONSTANT, 0);
  return ret;
}

Ball triangulate_ball(const Circle &circle_A, const Circle &circle_B,
                      const StereoCamera &stereo_camera) {
  const Camera &cam_A = stereo_camera.camera_A;
  const Camera &cam_B = stereo_camera.camera_B;

  Ball ball;

  // position
  Vector3d center;
  double distance = stereo_camera.triangulate(circle_A.location_px,
                                              circle_B.location_px, center);

  // radius
  double radius_A = cam_A.estimate_radius_world_m(center, circle_A.radius_px);
  double radius_B = cam_B.estimate_radius_world_m(center, circle_B.radius_px);

  // assert the radii do not disagree too much
  assert(abs(radius_A / radius_B - 1) < 0.3);

  double radius = (radius_A + radius_B) * 0.5;

  // better direction:
  // Construct Plane for each Camera through origin, ball center and direction
  // px

  auto pix_dir_A = circle_A.direction_px();
  auto loc_dir_A = cam_A.point_on_sensor_world(pix_dir_A);

  Eigen::Hyperplane<double, 3> plane_A = Eigen::Hyperplane<double, 3>::Through(
      cam_A.point_on_sensor_world(circle_A.location_px),
      cam_A.point_on_sensor_world(circle_A.direction_px()), cam_A.origin);
  Eigen::Hyperplane<double, 3> plane_B = Eigen::Hyperplane<double, 3>::Through(
      cam_B.point_on_sensor_world(circle_B.location_px),
      cam_B.point_on_sensor_world(circle_B.direction_px()), cam_B.origin);

  // use normals to find direction of intersection, which is all we need
  Eigen::Vector3d better_direction =
      plane_A.normal().cross(plane_B.normal()).normalized();

  ball.center_m = center;
  ball.radius_m = radius;
  ball.direction = better_direction;
  ball.confidence = 1 / (distance + 0.1);
  ball.connections_A = circle_A.connections;
  ball.connections_B = circle_B.connections;

  return ball;
}
