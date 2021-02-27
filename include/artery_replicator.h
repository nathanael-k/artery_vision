
#pragma once

#include <cstddef>
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

  static void update_display(int pos, void *ptr);
  static void change_visual(int pos, void *ptr);
  arteryGraph &build_graph();

  int display_source = 0;

private:
  const std::string window_A, window_B, window_detail;

  arteryGraph graph;

  std::list<Ball> generate_init_balls(size_t frame, size_t max_count);

  // takes a node, recursively explores what is around it and not yet registered
  // ends when having more than 2 connections in at least one camera, with a
  // discrepancy (A != B) in that case, still create the node, but add it to a
  // hot junction list

  // hot ends: ends that were added in this frame, we will need to check them in the next frame
  void explore_node(arteryNode &node, const arteryNode &old_node,
                    std::list<arteryNode *> hot_ends, 
                    std::list<arteryNode *> new_ends,
                    std::list<arteryNode *> hot_junctions, const size_t frame);
};