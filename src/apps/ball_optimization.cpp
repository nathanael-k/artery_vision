
#include <opencv2/highgui.hpp>

// datastructure
#include <arteryNet2.h>
// artery reconstruction algorithm
#include <artery_replicator.h>

#include <ostream>

// debug nan
#include <fenv.h>

// timings
#include <chrono>

int main(int argc, char **argv)
{
  auto t_start = std::chrono::high_resolution_clock::now();

  // fail on NaN or overflows... useful for debugging
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  std::string window_A = "Camera A";
  std::string window_B = "Camera B";
  std::string window_detail = "Detail";

  // create windows
  // cv::namedWindow( "Control");
  cv::namedWindow(window_A, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(window_B, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(window_detail, cv::WINDOW_AUTOSIZE);

  // Test scenarios:
  //ArteryReplicator replicator("../../../data/renders/easy_flow_3/", window_A, window_B, window_detail);
  //ArteryReplicator replicator("../../../data/renders/flow_1/", window_A, window_B, window_detail);

  // 3 phantom models:
  //ArteryReplicator replicator("../../../data/renders/aorta_to_brain/", window_A, window_B, window_detail);
  //ArteryReplicator replicator("../../../data/renders/heart/", window_A, window_B, window_detail);
  //ArteryReplicator replicator("../../../data/renders/real_brain/", window_A, window_B, window_detail);

  // Real x-ray:
  ArteryReplicator replicator("../../../data/real/png/", window_A, window_B, window_detail);

  // UI
  cv::createTrackbar("Frame:", "", &replicator.camera.current_displayed_frame,
                     replicator.camera.total_frames - 1, ArteryReplicator::update_display, &replicator);
  cv::createTrackbar("Vis. Src:", "", &replicator.display_source, 6, ArteryReplicator::change_visual, &replicator);

  auto t_preprocess = std::chrono::high_resolution_clock::now();

  // build graph
  arteryGraph &graph = replicator.build_graph();

  auto t_graph = std::chrono::high_resolution_clock::now();

  // write sphere location to file
  std::ofstream file;
  file.open("../data/out/graph.txt");

  write_to_file(graph, file);
  file.close();

  auto t_end = std::chrono::high_resolution_clock::now();

  double total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.;

  double preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(t_preprocess - t_start).count() / 1000.;
  double tt_graph = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph - t_preprocess).count() / 1000.;
  double output = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_graph).count() / 1000.;

  std::cout << "Total: " << total_duration << "s. Preprocess: " << preprocess << "s. Graph: " << tt_graph << "s Output: " << output << "s. Spheres: " << graph.size << ".\n";

  // exit after pressing any key
  cv::waitKey(0);

  return 0;
}