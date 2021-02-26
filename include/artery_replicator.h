
#pragma once

#include <string>

#include <arteryNet2.h>
#include <stereo_camera.h>

// replicates arteries
class ArteryReplicator {
public:
  StereoCamera camera;


  ArteryReplicator(const std::string &data_path, const std::string &window_A,
                   const std::string &window_B,
                   const std::string &window_detail);

  static void update_display(int pos, void* ptr);
   static void change_visual(int pos, void* ptr);
  arteryGraph &build_graph();

  int display_source = 0;

private:
  const std::string window_A, window_B, window_detail;

  arteryGraph graph;
};