#include <artery_replicator.h>

#include <imageData2.h>
#include <arteryNet2.h>
#include <cstddef>
#include <stereo_camera.h>

#include <ball_optimizer.h>

#include <opencv2/highgui.hpp>

ArteryReplicator::ArteryReplicator(const std::string &data_path,
                                   const std::string &window_A,
                                   const std::string &window_B,
                                   const std::string &window_detail)
    : camera(data_path), window_A(window_A), window_B(window_B),
      window_detail(window_detail)
{

  if (camera.image_data_A.source.empty() ||
      camera.image_data_B.source.empty())
  {
    std::cout << "Could not open or find the image!\n"
              << std::endl;
    assert(false);
  }
}

arteryGraph &ArteryReplicator::build_graph()
{

  change_visual(6, this);

  for (size_t frame = 0; frame < camera.total_frames; frame++)
  {
    camera.current_displayed_frame = frame;
    std::list<Ball> init_balls = generate_init_balls(frame, 20);

    std::list<Ball *> end_balls, lone_balls;

    // add end init balls to the graph
    for (auto &ball : init_balls)
    {
      if (ball.connections_A == 0 || ball.connections_B == 0)
      {
        lone_balls.emplace_back(&ball);
      }
      else if (ball.connections_A == 1 && ball.connections_B == 1)
      {
        end_balls.emplace_back(&ball);
      }
      else if (ball.connections_A >= 3 || ball.connections_B >= 3)
      {
        //junction_balls.emplace_back(&ball);
      }
      else if (ball.connections_A == 2 || ball.connections_B == 2)
      {
        //middle_balls.emplace_back(&ball);
      }
    }

    // we need at least one end ball to start the build
    //if (end_balls.size() == 0 && graph.all_node) {
    // skip build
    //}

    std::cout << "--- New Frame with " << end_balls.size()
              << " new ends to explore.\n";

    for (Ball *ball : end_balls)
    {
      start_at_end_ball(*ball, frame);
      camera.image_data_A.drawGraph(graph, frame);
      camera.image_data_B.drawGraph(graph, frame);
      update_display(0, this);
      //cv::waitKey();
    }

    auto restart_indices_copy = restart_indices;
    restart_indices.clear();
    for (const auto &idx : restart_indices_copy)
    {
      if (restart_at(idx, frame))
      {
        restart_indices.emplace_back(idx);
      }
      camera.image_data_A.drawGraph(graph, frame);
      camera.image_data_B.drawGraph(graph, frame);
      update_display(0, this);
      //cv::waitKey();
    }

    //cv::waitKey();
  }

  return graph;
}

void ArteryReplicator::update_display(int pos, void *ptr)
{
  auto *artery = static_cast<ArteryReplicator *>(ptr);

  cv::imshow(artery->window_A, artery->camera.displayed_image_A());
  cv::imshow(artery->window_B, artery->camera.displayed_image_B());
}

void ArteryReplicator::change_visual(int pos, void *ptr)
{
  std::string text = "Image source: ";
  from f;
  if (pos == 0)
  {
    f = from::source;
    text.append("source");
  }
  if (pos == 1)
  {
    f = from::threshold;
    text.append("threshold");
  }
  if (pos == 2)
  {
    f = from::initConv;
    text.append("initConv");
  }
  if (pos == 3)
  {
    f = from::distance;
    text.append("distance");
  }
  if (pos == 4)
  {
    f = from::endpoints;
    text.append("endpoints");
  }
  if (pos == 5)
  {
    f = from::buffer;
    text.append("buffer");
  }
  if (pos == 6)
  {
    f = from::visualisation;
    text.append("visualisation");
  }

  auto *artery = static_cast<ArteryReplicator *>(ptr);

  artery->camera.reset_visual(f);
  cv::displayStatusBar(artery->window_detail, text);
  update_display(0, ptr);
}

std::list<Ball> ArteryReplicator::generate_init_balls(size_t frame,
                                                      size_t max_count)
{
  // auto init circles:
  std::vector<Circle> init_Circles_A =
      extract_init_circles(max_count, camera.image_data_A.initConv[frame],
                           camera.image_data_A.threshold[frame],
                           camera.image_data_A.distance[frame]);

  std::vector<Circle> init_Circles_B =
      extract_init_circles(max_count, camera.image_data_B.initConv[frame],
                           camera.image_data_B.threshold[frame],
                           camera.image_data_B.distance[frame]);

  // cross correlate
  auto distances =
      cross_correlate_circles(init_Circles_A, init_Circles_B, camera);

  std::cout << distances << std::endl;

  // based on cross correlation, build balls (this list owns the Balls)
  return init_balls(init_Circles_A, init_Circles_B, camera);
}

// tail recursive!
void ArteryReplicator::explore_node(size_t node_idx, size_t old_node_idx,
                                    const size_t frame)
{
  // 0) the old_node is already optimized, the node not yet
  // we know that there is no node yet at the center of the new node!

  while (true)
  {
    arteryNode &node = graph.all_nodes[node_idx];
    const arteryNode &old_node = graph.all_nodes[old_node_idx];
    ;

    Ball &ball = node.ball;
    BallOptimizer optimizer(ball, camera);

    // 1) find all neighbours we have

    // far circles help us understand what the situation is a bit further out.
    // it improves probability that we have the same number of connections on
    // both cameras
    auto far_circles_A = optimizer.report_adjacent_circles(false, 2, frame);
    auto far_circles_B = optimizer.report_adjacent_circles(true, 2, frame);
    ball.connections_A = far_circles_A.size();
    ball.connections_B = far_circles_B.size();
    // TODO: expand to what we might do if we dont agree
    // assert(ball.connections_A == ball.connections_B);
    size_t max_connections = std::max(ball.connections_A, ball.connections_B);
    size_t min_connections = std::min(ball.connections_A, ball.connections_B);

    /*
    if (max_connections == 0) {
      // that is indeed strange and should not really happen, probably the graph
      // is not connected or maybe the thresholding is too aggressive
      //return;
      assert(false);
    }*/

    if (min_connections == 0)
    {
      return;
    }

    // if we have one connection on both, it seems to be a dead end
    if (max_connections == 1)
    {
      optimizer.optimize_constrained(10, frame, old_node.ball, 1.5);
      restart_indices.emplace_back(node_idx);
      std::cout << "Ending line at new found ending ..." << std::endl;
      return;
    }

    // if we have a 1 and a 3, the probability that 3 is a mistake is high
    if (min_connections == 1 && max_connections >= 3)
    {
      optimizer.optimize_constrained(10, frame, old_node.ball, 1.5);
      restart_indices.emplace_back(node_idx);
      std::cout << "Ending line at new found strange position ..." << std::endl;
      return;
    }

    // if at least one of the circles is 3 - connected, we have probably found a
    // junction so lets optimize a junction and then add this junction as a
    // special node
    if (min_connections >= 3)
    {

      // just optimize this one, then leave
      std::cout << "Ending line at new found junction ..." << std::endl;

      optimizer.optimize_junction(10, frame);
      return;
    }

    assert(ball.connections_A == 2 || ball.connections_B == 2);
    // the normal path case, figure out which of the circles is not the last
    // ball then we can create a new ball just there

    // add that easy line ball:

    optimizer.optimize_constrained(50, frame, old_node.ball, 1.5);
    std::cout << "Continuing line ..." << std::endl;

    // go for the next ball
    auto circles_A = optimizer.report_adjacent_circles(false, 1.5, frame);
    auto circles_B = optimizer.report_adjacent_circles(true, 1.5, frame);
    bool is_prob_junction = (circles_A.size() > 2 || circles_B.size() > 2);

    Circle last_circle_A = project_circle(old_node.ball, camera.camera_A);
    Circle last_circle_B = project_circle(old_node.ball, camera.camera_B);

    // nice trick to fix things if connections are not the same for both
    circles_A.emplace_back(project_circle(ball, camera.camera_A));
    circles_B.emplace_back(project_circle(ball, camera.camera_B));

    Circle next_circle_A = find_furthest_circle(circles_A, last_circle_A);
    Circle next_circle_B = find_furthest_circle(circles_B, last_circle_B);

    Ball next_ball = triangulate_ball(next_circle_A, next_circle_B, camera);
    size_t next_index;

    // if we already have a ball in the graph at the same place, connect them
    // CONNECTION RADIUS must be choosen small enough to avoid spurious balls
    double connect_radius = is_prob_junction ? 1.6 : 0.9; // rendered
    // double connect_radius = is_prob_junction ? 2.0 : 1.2; // printed

    if (graph.find_closest_ball(next_ball.center_m, node.index, next_index) <
        connect_radius)
    {
      // connect them, move on

      graph.connectNodes(node.index, next_index);
      node.explored_index = frame;
      graph.all_nodes[next_index].explored_index = frame;

      std::cout << "Connecting already found nodes " << node.index << " and "
                << next_index << '\n';
      return;
    }
    if (next_ball.confidence < 2)
    {
      std::cout << "Rejecting next ball due to confidence of " << next_ball.confidence << " after idx "
                << node_idx << '\n';
      return;
    }

    node.explored_index = frame;
    // references to nodes get invalidated here!
    size_t next_node_idx = graph.add_ball_at(next_ball, node_idx);

    // instead of recursion
    old_node_idx = node_idx;
    node_idx = next_node_idx;
  }
}

void ArteryReplicator::start_at_end_ball(Ball &ball, const size_t frame_index)
{

  // first make sure there is no node yet in the graph at that position
  size_t index;
  // short circuit trick to not consider empty graphs
  if (!graph.all_nodes.empty() &&
      graph.find_closest_ball(ball.center_m, index) < 1)
  {
    std::cout << "End Ball was already part of the graph.\n";
    return;
  }

  // add this ball to the graph, the next ball should be an obvious one
  size_t node_idx = graph.add_ball(ball);
  restart_indices.emplace_back(node_idx);

  // optimize the ball
  BallOptimizer optimizer(graph.all_nodes[node_idx].ball, camera);

  // do the whole triangulation business of the next ball

  // making sure that we have only one next ball
  double radius_A = 1.5;
  auto circles_A =
      optimizer.report_adjacent_circles(false, radius_A, frame_index);
  while (circles_A.size() > 1)
  {
    radius_A += 0.1;
    circles_A = optimizer.report_adjacent_circles(false, radius_A, frame_index);
  }

  double radius_B = 1.5;
  auto circles_B =
      optimizer.report_adjacent_circles(true, radius_B, frame_index);
  while (circles_B.size() > 1)
  {
    radius_B += 0.1;
    circles_B = optimizer.report_adjacent_circles(true, radius_B, frame_index);
  }

  if (circles_A.size() == 0 || circles_B.size() == 0)
  {
    return;
  }

  assert(circles_A.size() == 1 && circles_B.size() == 1);

  // optimizer.optimize(10, frame_index);

  size_t next_node_idx = graph.add_ball_at(
      triangulate_ball(circles_A[0], circles_B[0], camera), node_idx);

  explore_node(next_node_idx, node_idx, frame_index);
}

// returns true if the node should stay in the list
bool ArteryReplicator::restart_at(const size_t node_idx,
                                  const size_t frame_index)
{
  // only restart if we have just degree 1 right now, but we show 2 circles in
  // both cameras
  arteryNode &node = graph.all_nodes[node_idx];
  if (node.degree != 1)
    return false;

  // check vicinity:
  BallOptimizer optimizer(graph.all_nodes[node_idx].ball, camera);
  auto circles_A = optimizer.report_adjacent_circles(false, 2, frame_index);
  auto circles_B = optimizer.report_adjacent_circles(true, 2, frame_index);
  if (circles_A.size() != 2 || circles_B.size() != 2)
    return true;

  // find out which of the circles is furthest away from already connected node
  arteryNode &other_node = graph.all_nodes[node.paths[0]];

  Circle last_circle_A = project_circle(other_node.ball, camera.camera_A);
  Circle last_circle_B = project_circle(other_node.ball, camera.camera_B);

  auto next_circle_A = find_furthest_circle(circles_A, last_circle_A);
  auto next_circle_B = find_furthest_circle(circles_B, last_circle_B);

  Ball next_ball = triangulate_ball(next_circle_A, next_circle_B, camera);
  if (next_ball.confidence < 1)
  {
    std::cout << "Next ball after restart had too high discrepancy.\n";
    return false;
  }

  if (next_circle_A.radius_px < 0.5 * last_circle_A.radius_px ||
      next_circle_B.radius_px < 0.5 * last_circle_B.radius_px)
  {
    std::cout << "Next ball after restart had too small circles.\n";
    return false;
  }

  if (next_circle_A.radius_px > 2 * last_circle_A.radius_px ||
      next_circle_B.radius_px > 2 * last_circle_B.radius_px)
  {
    std::cout << "Next ball after restart had too big circles.\n";
    return false;
  }

  BallOptimizer next_optimizer(next_ball, camera);
  next_optimizer.optimize_constrained(10, frame_index, node.ball, 1.5);

  size_t next_index;

  // if we already have a ball in the graph at the same place, connect them

  if (graph.find_closest_ball(next_ball.center_m, next_index) < 0.9)
  {
    // connect them, move on
    graph.connectNodes(node.index, next_index);
    node.explored_index = frame_index;
    graph.all_nodes[next_index].explored_index = frame_index;

    std::cout << "Connecting already found nodes (at restart) " << node.index
              << " and " << next_index << '\n';
    return false;
  }

  node.explored_index = frame_index;
  // references to nodes get invalidated here!
  size_t next_node_idx = graph.add_ball_at(next_ball, node_idx);

  std::cout << "Restart from node " << node_idx << "added nodes.\n";
  explore_node(next_node_idx, node_idx, frame_index);

  return false;
}