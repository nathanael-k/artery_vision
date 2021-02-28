
#include "opencv2/highgui.hpp"
#include <arteryNet2.h>
#include <artery_replicator.h>

#include <ostream>

// debug nan
#include <fenv.h>

int main(int argc, char **argv) {

  feenableexcept(FE_INVALID | FE_OVERFLOW);

  std::string window_A = "Camera A";
  std::string window_B = "Camera B";
  std::string window_detail = "Detail";

  // create windows
  // cv::namedWindow( "Control");
  cv::namedWindow(window_A, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(window_B, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(window_detail, cv::WINDOW_AUTOSIZE);

  
  //ArteryReplicator replicator("../data/renders/easy_flow_3/", window_A, window_B, window_detail);

  //ArteryReplicator replicator("../data/renders/aorta_to_brain/", window_A, window_B, window_detail);
//ArteryReplicator replicator("../data/renders/flow_1/", window_A, window_B, window_detail);
ArteryReplicator replicator("../data/renders/real_brain/", window_A, window_B, window_detail);

  cv::createTrackbar("Frame:", "", &replicator.camera.current_displayed_frame,
                     replicator.camera.total_frames - 1, ArteryReplicator::update_display, &replicator);
  cv::createTrackbar("Vis. Src:", "", &replicator.display_source, 6, ArteryReplicator::change_visual, &replicator);
  

  arteryGraph& graph = replicator.build_graph();
  
  std::ofstream file;
  file.open("../data/out/graph.txt");

  write_to_file(graph, file);
  file.close();

  cv::waitKey(0);

  return 0;
}