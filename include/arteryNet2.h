#ifndef ARTERYNET_ARTERY_NET_H
#define ARTERYNET_ARTERY_NET_H

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
    bool optimized = false;
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

    // adds a node to the graph, connected to this node
    arteryNode& addNode(const Ball& ball);
};

class arteryGraph{
public:
    std::vector<arteryNode> all_nodes;

    arteryNode& root = all_nodes[0];
    // arteryNode* catheter_location = {nullptr};
    std::vector<size_t> junctions;

    // start a new net with some position as first node

    arteryGraph(const Ball& ball);
    // arteryGraph() {};

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
};


  // write graph to file
  void write_to_file(const arteryGraph &graph, std::ofstream&handle);

#endif //ARTERYNET_ARTERY_NET_H
