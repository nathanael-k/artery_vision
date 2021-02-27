#ifndef ARTERYNET_ARTERY_NET_H
#define ARTERYNET_ARTERY_NET_H

#include <cstddef>
#define MAX_DEGREE 10
#define DEFAULT_RADIUS 0.25
#include <vector>

#include <Eigen/Dense>
#include <ball.h>

class arteryGraph;

class arteryNode{
public:
    arteryGraph& graph;
    int degree = 0;
    // have we fully explored / explained this node
    int explored_index = -1;
    bool enddraw = true;
    const size_t index;

    Ball ball;

    // pointer to skip nodes: goes along the same path, but directly to the next special node
    size_t junctions[MAX_DEGREE];
    // the paths going from this node
    size_t paths[MAX_DEGREE];


    double radii[MAX_DEGREE];

    // construct a new arteryNode
    arteryNode(arteryGraph& _graph, const Ball& ball);

};

class arteryGraph{
public:
    std::vector<arteryNode> all_nodes;

    // arteryNode* catheter_location = {nullptr};
    std::vector<size_t> junctions;

    // start a new net with some position as first node
    arteryGraph(const Ball& ball);
    arteryGraph() {};

    // establish direct link between junctions for faster processing
    // void contractPath(arteryNode* start, int direction);

    // two nodes that should get connected in the graph
    void connectNodes(size_t index_1, size_t index_2);

    // enable direct links between nodes with degree != 2
    // establish root node with degree >2
    //void optimize();
    int size = 0;
    int nEnd = 0;
    int nJunction = 0;
    bool optimized = false;

    // distance and index of the closest ball
    // distance is calculated relative to the radius of that ball,
    // 1.0 is exactly on the surface of the ball
    double find_closest_ball(const Eigen::Vector3d position_m, size_t& index) const;
    
    // adds a new, unconnected ball to the graph (typically a not yet connected end ball)
    size_t add_ball(const Ball& ball);

    // adds a node to the graph, connected to this node
    size_t add_ball_at (const Ball& ball, const size_t index);
};


  // write graph to file
  void write_to_file(const arteryGraph &graph, std::ofstream&handle);

#endif //ARTERYNET_ARTERY_NET_H
