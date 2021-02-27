#include <arteryNet2.h>

#include <assert.h>
#include <cstddef>
#include <fstream>
#include <limits>

arteryNode::arteryNode(arteryGraph& _graph, const Ball& ball)
                    : graph(_graph), ball(ball), index(graph.size) {
    graph.nEnd++;
    graph.size++;
}

size_t arteryGraph::add_ball( const Ball& ball) {
    size_t index = all_nodes.size();
    all_nodes.emplace_back(*this, ball);
    return index;
}

size_t arteryGraph::add_ball_at(const Ball& ball, const size_t index) {

    assert(all_nodes[index].degree < MAX_DEGREE);

    size_t new_index = all_nodes.size();
    all_nodes.emplace_back(*this, ball);
    arteryNode& new_node = all_nodes.back();
    arteryNode& old_node = all_nodes[index];

    // connections
    old_node.paths[old_node.degree] = new_index;
    old_node.degree++;
    new_node.paths[new_node.degree] = index;
    new_node.degree++;

    // can only happen once
    if (old_node.degree == 3) nJunction++;
    if (old_node.degree == 2) nEnd--;

    return new_index;
}

arteryGraph::arteryGraph(const Ball& ball) {
    all_nodes.emplace_back(*this, ball);
}

void arteryGraph::connectNodes(size_t index_1, size_t index_2) {
    arteryNode& node_1 = all_nodes[index_1];
    arteryNode& node_2 = all_nodes[index_2];
    

    assert( node_1.degree < MAX_DEGREE &&
            node_2.degree < MAX_DEGREE);

    // update the old node
    node_1.paths[node_1.degree] = index_2;
    node_1.degree++;

    // update the new node
    node_2.paths[node_2.degree] = index_1;
    node_2.degree++;

    if (node_1.degree == 2)
        nEnd--;
    if (node_2.degree == 2)
        nEnd--;
    if (node_1.degree == 3)
        nJunction++;
    if (node_2.degree == 3)
        nJunction++;
}

/*
void arteryGraph::optimize() {
    assert(optimized == false);
    optimized = true;

    // make sure we start somewhere that is already deg != 2
    assert( root->degree != 2);
    junctions.clear();
    junctions.push_back(root);

    // iterate over the vector with all junctions in the graph
    for (int i = 0; i < junctions.size(); i++) {
        assert (junctions[i]->degree != 2);
        // for each junction, go trough all connections
        for (int j = 0; j < junctions[i]->degree; j++) {
            if (junctions[i]->junctions[j] == nullptr) {
                contractPath(junctions[i], j);
            }
        }
    }
}
*/
/*
void arteryGraph::contractPath(arteryNode* start, int direction) {
    // make sure we are save
    assert(start->degree > direction);
    assert(start->paths[direction] != nullptr);
    assert(start->junctions[direction] == nullptr);

    arteryNode* current = start->paths[direction];
    arteryNode* last = start;

    // path forward index is an invariant on paths
    int forward_direction = 0;
    if (current->paths[forward_direction]==start)
        forward_direction = 1;

    double min_radius = current->ball.radius_m;

    while (current->degree == 2) {
        // make sure we never track back
        if (current->paths[forward_direction]->degree ==2)
            assert(current->paths[forward_direction]->paths[forward_direction] != current);

        last = current;
        current = current->paths[forward_direction];

        min_radius = std::min(min_radius, current->ball.radius_m);
    }

    // we are at another junction (degree != 2)
    start->junctions[direction] = current;
    start->radii[direction] = min_radius;

    // find direction at current to start
    for (int i = 0; i < current->degree; i++) {
        if (current->paths[i] == last) {
            current->junctions[i] = start;
            current->radii[i] = min_radius;
        }
    }

    // if our junction is not yet added we add it to get it checked later
    if (current->optimized == false)
        junctions.push_back(current);
}
*/
  // write graph to file
  void write_to_file(const arteryGraph &graph, std::ofstream&handle) {
      for (const auto& node : graph.all_nodes)
      {
          handle << node.ball.center_m.x() << " " << node.ball.center_m.y() << " "
           << node.ball.center_m.z() << " " << node.ball.radius_m << '\n';
      }
  }

  double arteryGraph::find_closest_ball(const Eigen::Vector3d position_m, size_t& index) const {
      double min_distance = std::numeric_limits<double>::max();
      assert(!all_nodes.empty());
      for (const arteryNode& node : all_nodes) {
          double rel_distance = (position_m - node.ball.center_m).norm() / node.ball.radius_m;
          if (rel_distance < min_distance) {
              min_distance = rel_distance;
              index = node.index;
          }
      }
      return min_distance;
  }
