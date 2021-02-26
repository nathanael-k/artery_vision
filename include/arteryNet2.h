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
    const int index;

    Ball ball;

    // pointer to skip nodes: goes along the same path, but directly to the next special node
    arteryNode* junctions[MAX_DEGREE] = {nullptr};
    // the paths going from this node
    arteryNode* paths[MAX_DEGREE] = {nullptr};


    double radii[MAX_DEGREE];

    // construct a new arteryNode
    arteryNode(arteryGraph& _graph, const Ball& ball);

    // adds a node to the graph, connected to this node
    arteryNode* addNode(const Ball& ball);
};

class arteryGraph{
public:
    arteryNode* root = {nullptr};
    arteryNode* catheter_location = {nullptr};
    std::vector<arteryNode*> junctions;

    // start a new net with some position as first node

    arteryGraph(const Ball& ball);
    arteryGraph() {};

    // establish direct link between junctions for faster processing
    void contractPath(arteryNode* start, int direction);

    // two nodes that should get connected in the graph
    void connectNodes(arteryNode* node_a, arteryNode* node_b);

    // enable direct links between nodes with degree != 2
    // establish root node with degree >2
    void optimize();
    int size = 0;
    int nEnd = 0;
    int nJunction = 0;
    bool optimized = false;
};

 // write everything connected to that node to a file for further processing
  void write_to_file(const arteryNode &node, std::ofstream &handle);

#endif //ARTERYNET_ARTERY_NET_H
