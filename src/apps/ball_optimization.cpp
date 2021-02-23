#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <ball_optimizer.h>
#include <imageData2.h>
#include <ostream>
#include <stereo_camera.h>

int kernel_radius = 11;

StereoCamera camera("../data/renders/aorta_to_brain/");

// imageData data1("../data/renders/aorta_to_brain/", 0),
//          data2("../data/renders/aorta_to_brain/", 1);

void displayVisual(int = 0, void * = nullptr) {
  // imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
  imshow("Cam1 Visual", camera.displayed_image_A());
  imshow("Cam2 Visual", camera.displayed_image_B());
  // imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
}

void changeVisual(int pos, void *) {

  std::string text = "Image source: ";
  from f;
  if (pos == 0) {
    f = from::source;
    text.append("source");
  }
  if (pos == 1) {
    f = from::threshold;
    text.append("threshold");
  }
  if (pos == 2) {
    f = from::initConv;
    text.append("initConv");
  }
  if (pos == 3) {
    f = from::distance;
    text.append("distance");
  }
  if (pos == 4) {
    f = from::endpoints;
    text.append("endpoints");
  }
  if (pos == 5) {

    f = from::buffer;
    text.append("buffer");
  }
  if (pos == 6) {
    f = from::visualisation;
    text.append("visualisation");
  }

  camera.reset_visual(f);
  cv::displayStatusBar("Kernel", text);
  displayVisual(0, 0);
}

void thresholdCurrent(int state, void *) {
  // if button got pressed down
  if (state == 0) {
    camera.image_data_A.apply_threshold(camera.current_displayed_frame, 128,
                                        255);
    camera.image_data_B.apply_threshold(camera.current_displayed_frame, 128,
                                        255);
    displayVisual();
  }
}

void applyInitKernel(int state, void *) {
  cv::filter2D(camera.image_data_A.threshold[camera.current_displayed_frame],
               camera.image_data_A.initConv[camera.current_displayed_frame], -1,
               circleKernel(kernel_radius));
  cv::filter2D(camera.image_data_B.threshold[camera.current_displayed_frame],
               camera.image_data_B.initConv[camera.current_displayed_frame], -1,
               circleKernel(kernel_radius));

  camera.image_data_A.initConv[camera.current_displayed_frame].convertTo(
      camera.image_data_A.initConv[camera.current_displayed_frame], CV_8U, 255);
  camera.image_data_B.initConv[camera.current_displayed_frame].convertTo(
      camera.image_data_B.initConv[camera.current_displayed_frame], CV_8U, 255);

  displayVisual();
}

int main(int argc, char **argv) {

  if (camera.image_data_A.source.empty() ||
      camera.image_data_B.source.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    return -1;
  }

  // create windows
  // cv::namedWindow( "Control");
  cv::namedWindow("Cam1 Visual", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Cam2 Visual", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Kernel", cv::WINDOW_AUTOSIZE);
  displayVisual();

  int what = 0;

  cv::createButton("Threshold", thresholdCurrent);
  cv::createButton("Apply Kernel", applyInitKernel);
  cv::createTrackbar("Frame:", "", &camera.current_displayed_frame,
                     camera.total_frames - 1, displayVisual);
  cv::createTrackbar("Vis. Src:", "", &what, 6, changeVisual);
  cv::createTrackbar("Kernel Size", "", &kernel_radius, 21, nullptr);

  // auto init circles:
  std::vector<Circle> init_Circles_A = extract_init_circles(
      5, camera.image_data_A.initConv[0], camera.image_data_A.threshold[0],
      camera.image_data_A.distance[0]);

  std::vector<Circle> init_Circles_B = extract_init_circles(
      5, camera.image_data_B.initConv[0], camera.image_data_B.threshold[0],
      camera.image_data_B.distance[0]);
  

  // cross correlate
  auto distances = cross_correlate_circles(init_Circles_A, init_Circles_B, camera);

  std::cout << distances << std::endl;

  // based on cross correlation, build balls
  auto balls = init_balls(init_Circles_A, init_Circles_B, camera);

  std::vector<Ball> end_balls, middle_balls, junction_balls;
  // sort balls into bins
  for (const auto& ball : balls){
    if (ball.connections_A == 0 || ball.connections_B == 0) {
      // skip
    }
    else if (ball.connections_A == 1 && ball.connections_B == 1) {
      end_balls.emplace_back(ball);
    }
    else if(ball.connections_A >= 3 ||  ball.connections_B >=3) {
      junction_balls.emplace_back(ball);
    }
    else if(ball.connections_A == 2 || ball.connections_B == 2) {
      middle_balls.emplace_back(ball);
    }
  }


  // pick the first ball and start to build line from there
  Ball proto_ball = end_balls[0];
  

  // optimize first ball (with connections 1)
  BallOptimizer optimizer(proto_ball, camera);
  optimizer.optimize(10, 0);

  // find next ball
  Ball next_ball = proto_ball.next_ball();

  BallOptimizer next(next_ball, camera);
  optimizer.optimize_constrained(10, 0, proto_ball, 1.5);

  // until we are inside an existing end ball


  
  // initial maximal circles:
  Circle init_ball_A(Eigen::Vector2d(609, 924), 8, 0);
  Circle init_ball_B(Eigen::Vector2d(485, 959), 7, 0);

  cv::waitKey(0);


  return 0;
}