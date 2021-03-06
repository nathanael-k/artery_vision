#include <Eigen/Dense>
#include <cstddef>
#include <stereo_camera.h>
#include <ball_optimizer.h>
#include <list>

StereoCamera::StereoCamera(std::string meta_folder)
    : camera_A(meta_folder + "meta", 0), camera_B(meta_folder + "meta", 1),
      image_data_A(meta_folder, camera_A), image_data_B(meta_folder, camera_B) {
  total_frames = image_data_A.size;
  assert(image_data_A.size == image_data_B.size);
}

double StereoCamera::triangulate(const Eigen::Vector2d &point_cam_A,
                                 const Eigen::Vector2d &point_cam_B,
                                 Eigen::Vector3d &out_intersection) const {
  // rely on camera implementation
  return Camera::intersect(camera_A, point_cam_A, camera_B, point_cam_B,
                           out_intersection);
}

const imageData &StereoCamera::image_data(uint8_t index) const {
  if (index == 0)
    return image_data_A;
  if (index == 1)
    return image_data_B;

  // invalid index
  assert(false);
}

const cv::Mat &StereoCamera::displayed_image_A() const {
  return (*image_data_A.curr_displayed)[current_displayed_frame];
}

const cv::Mat &StereoCamera::displayed_image_B() const {
  return (*image_data_B.curr_displayed)[current_displayed_frame];
}

void StereoCamera::reset_visual(from where) {
  image_data_A.resetVisual(where, current_displayed_frame);
  image_data_B.resetVisual(where, current_displayed_frame);
}

Eigen::MatrixXd StereoCamera::cross_correlate_pixels(
    const std::vector<Eigen::Vector2d> &pixel_A,
    const std::vector<Eigen::Vector2d> &pixel_B) const {
  assert(pixel_A.size() > 0 && pixel_B.size() > 0);
  Eigen::MatrixXd ret = Eigen::MatrixXd(pixel_A.size(), pixel_B.size());

  Eigen::Vector3d throwaway;

  for (int i = 0; i < pixel_A.size(); i++) {
    for (int j = 0; j < pixel_B.size(); j++) {
      ret(i, j) = triangulate(pixel_A[i], pixel_B[j], throwaway);
    }
  }

  return ret;
}

Eigen::MatrixXd cross_correlate_circles(const std::vector<Circle> &circles_A,
                                        const std::vector<Circle> &circles_B,
                                        const StereoCamera &camera) {
  size_t dim_A, dim_B;
  dim_A = circles_A.size();
  dim_B = circles_B.size();
  assert(dim_A > 0 && dim_B > 0);

  std::vector<Eigen::Vector2d> pixel_A, pixel_B;
  for (const auto &circle : circles_A)
    pixel_A.emplace_back(circle.location_px);
  for (const auto &circle : circles_B)
    pixel_B.emplace_back(circle.location_px);

  // for now, just raw distances
  return camera.cross_correlate_pixels(pixel_A, pixel_B);
}

std::list<Ball> init_balls(const std::vector<Circle> &circles_A,
                             const std::vector<Circle> &circles_B,
                             const StereoCamera &camera) {
  // cross correlate to find distances
  auto distances = cross_correlate_circles(circles_A, circles_B, camera);

  std::list<Ball> ret;
  size_t count = std::min(circles_A.size(), circles_B.size());
  double max = distances.maxCoeff();
  double cutoff = 0.15;

 for (int i = 0; i < count; i++) {
      Eigen::MatrixXd::Index minRow, minCol;
      double distance = distances.minCoeff(&minRow, &minCol);
      if (distance > 0.1 && distance > cutoff) // if we get one too bad we abort
        return ret;
      cutoff =  distance * 2; 
      ret.emplace_back(triangulate_ball(circles_A[minRow], circles_B[minCol], camera));
      distances.row(minRow).setConstant(max);
      distances.col(minCol).setConstant(max);
  }

  return ret;
}

Circle project_circle(const Ball& ball, const Camera& cam) {
  
  Circle ret = {cam.projectPoint(ball.center_m),
                cam.estimate_radius_image_px(ball.center_m, ball.radius_m), 0};

  Eigen::Vector2d A_direction_px =
      cam.projectPoint(ball.center_m + ball.direction) - ret.location_px;
  ret.set_angle_rad(atan2(A_direction_px.x(), -A_direction_px.y()));
  return ret;
}