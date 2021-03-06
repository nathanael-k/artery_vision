
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

// at interesting nodes we want to make a restart every frame
  std::vector<size_t> restart_indices;

// private:
  const std::string window_A, window_B, window_detail;

  arteryGraph graph;

  std::list<Ball> generate_init_balls(size_t frame, size_t max_count);

  // takes a node, recursively explores what is around it and not yet registered
  // ends when having more than 2 connections in at least one camera, with a
  // discrepancy (A != B) in that
  void explore_node(size_t node, size_t old_node, const size_t frame);

  void start_at_end_ball(Ball& ball, const size_t frame_index);

// we restart every frame from some interesting nodes
  bool restart_at(const size_t node_idx, const size_t frame_index);
  };