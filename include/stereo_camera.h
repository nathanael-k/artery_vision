#pragma once

#include <camera.h>
#include <imageData2.h>

class StereoCamera {
public:
  const Camera camera_A;
  const Camera camera_B;
  imageData image_data_A;
  imageData image_data_B;

  int current_displayed_frame = 0;
  int total_frames;

  const imageData &image_data(uint8_t index) const;

  const cv::Mat &displayed_image_A() const;
  const cv::Mat &displayed_image_B() const;

  void reset_visual(from where);
  // arteryGraph artery_graph;

  StereoCamera(std::string meta_folder);

  // given pixel coordinates for both cameras, find the 3d point
  double triangulate(const Eigen::Vector2d &point_cam_A,
                     const Eigen::Vector2d &point_cam_B,
                     Eigen::Vector3d &out_vector) const;

  // given mutiple coordinates, generate a distance matrix
  Eigen::MatrixXd
  cross_correlate_pixels(const std::vector<Eigen::Vector2d> &pixel_A,
                         const std::vector<Eigen::Vector2d> &pixel_B) const;
};

Eigen::MatrixXd cross_correlate_circles(const std::vector<Circle> &circles_A,
                                        const std::vector<Circle> &circles_B,
                                        const StereoCamera &camera);

std::vector<Ball> init_balls(const std::vector<Circle> &circles_A,
                                        const std::vector<Circle> &circles_B,
                                        const StereoCamera &camera);

Circle project_circle(const Ball&, const Camera& cam);