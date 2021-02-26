#include "ball.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <arteryNet2.h>

#include <ball_optimizer.h>
#include <cstddef>
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
    // camera.image_data_A.apply_threshold(camera.current_displayed_frame, 128,
    //                            255);
    // camera.image_data_B.apply_threshold(camera.current_displayed_frame, 128,
    //                           255);
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

  int frame = 5;

  // auto init circles:
  std::vector<Circle> init_Circles_A =
      extract_init_circles(5, camera.image_data_A.initConv[frame],
                           camera.image_data_A.threshold[frame],
                           camera.image_data_A.distance[frame]);

  std::vector<Circle> init_Circles_B =
      extract_init_circles(5, camera.image_data_B.initConv[frame],
                           camera.image_data_B.threshold[frame],
                           camera.image_data_B.distance[frame]);

  // cross correlate
  auto distances =
      cross_correlate_circles(init_Circles_A, init_Circles_B, camera);

  std::cout << distances << std::endl;

  // based on cross correlation, build balls (this list owns the Balls)
  std::list<Ball> balls = init_balls(init_Circles_A, init_Circles_B, camera);

  std::list<Ball*> end_balls, middle_balls, junction_balls;
  std::list<Ball*> hot_endings;
  // sort balls into bins
  for (auto &ball : balls) {
    if (ball.connections_A == 0 || ball.connections_B == 0) {
      // skip
    } else if (ball.connections_A == 1 && ball.connections_B == 1) {
      end_balls.emplace_back(&ball);
    } else if (ball.connections_A >= 3 || ball.connections_B >= 3) {
      junction_balls.emplace_back(&ball);
    } else if (ball.connections_A == 2 || ball.connections_B == 2) {
      middle_balls.emplace_back(&ball);
    }
  }

  assert(end_balls.size() > 0);

  // pick the first ball and start to build line from there
  Ball& pre_ball = *end_balls.front();
  end_balls.pop_front();


  // optimize first ball (with connections 1)
  BallOptimizer optimizer(pre_ball, camera);
  optimizer.optimize(10, frame);

  arteryGraph graph(pre_ball);

  arteryNode *curr_node = graph.root;
  hot_endings.emplace_back(&(curr_node->ball));
  Circle last_circle_A = project_circle(curr_node->ball, camera.camera_A);
  Circle last_circle_B = project_circle(curr_node->ball, camera.camera_B);

  arteryNode &root = *curr_node;
  arteryNode *next_node = nullptr;

  // we know we start at an end, so just proceed to next
  Ball next_ball = curr_node->ball.next_ball();

  while (true) {
    camera.image_data_A.drawGraph(root, frame);
    camera.image_data_B.drawGraph(root, frame);
    displayVisual();
    cv::waitKey();

    // classify next ball
    BallOptimizer next(next_ball, camera);
    auto Circles_A = next.report_adjacent_circles(false, 2, frame);
    auto Circles_B = next.report_adjacent_circles(true, 2, frame);
    size_t connections = std::max(Circles_A.size(), Circles_B.size());
    next_ball.connections_A = Circles_A.size();
    next_ball.connections_B = Circles_B.size();

    if (connections == 0) {
      // that is indeed strange and should not really happen, probably the graph
      // is not connected or maybe the thresholding is too aggressive
      assert(false);
    }

    // if we have one connection on both, it seems to be a dead end
    // we should check if we already found that, and then end this path
    if (connections == 1) {
      auto it = find_ball_at(end_balls, next_ball.center_m);
      if (it != end_balls.end()) {
        next_ball = **it;
        end_balls.erase(it);
        next.optimize(10, frame);

        std::cout << "Ending line at pre registered ending ..." << std::endl;
      }
      else {
      next.optimize_constrained(10, frame, pre_ball, 1.5);

        std::cout << "Ending line at new found ending ..." << std::endl;
      }
      next_node = curr_node->addNode(next_ball);
      hot_endings.emplace_back(&(next_node->ball));
      curr_node = next_node;      
      break;
    }

    // the normal path case, figure out which of the circles is not the last
    // ball then we can create a new ball just there
    if (connections == 2) {
      // add that easy line ball:

      next.optimize_constrained(10, frame, pre_ball, 1.5);
      std::cout << "Continuing line ..." << std::endl;
      next_node = curr_node->addNode(next_ball);
      curr_node = next_node;

      // go for the next ball
      Circles_A = next.report_adjacent_circles(false, 1.5, frame);
      Circles_B = next.report_adjacent_circles(true, 1.5, frame);

      // nice trick to fix things if connections are not the same for both
      Circles_A.emplace_back(project_circle(next_ball, camera.camera_A));
      Circles_B.emplace_back(project_circle(next_ball, camera.camera_B));

      Circle next_circle_A = find_furthest_circle(Circles_A, last_circle_A);
      Circle next_circle_B = find_furthest_circle(Circles_B, last_circle_B);

      pre_ball = next_ball;
      next_ball = triangulate_ball(next_circle_A, next_circle_B, camera);
    }

    // if at least one of the circles is 3 - connected, we have probably found a
    // junction so lets optimize a junction and then add this junction as a
    // special node
    if (connections == 3) {
      if (!junction_balls.empty()) {
      auto it = find_ball_at(junction_balls, next_ball.center_m);
      if (it != end_balls.end()) {
        next_ball = **it;
        junction_balls.erase(it);

        std::cout << "Ending line at pre registered junction ..." << std::endl;
      }
      else {

        std::cout << "Ending line at new found junction ..." << std::endl;
      }}
      else {
      std::cout << "Ending line at new found junction ..." << std::endl;
      }
      next.optimize_junction(10, frame);
      next_node = curr_node->addNode(next_ball);
      curr_node = next_node;      
      break;
    }
  }

  camera.image_data_A.drawGraph(root, frame);
  camera.image_data_B.drawGraph(root, frame);
  displayVisual();

  std::ofstream file;
  file.open("../data/out/graph.txt");

  write_to_file(root, file);
  file.close();

  cv::waitKey(0);

  return 0;
}