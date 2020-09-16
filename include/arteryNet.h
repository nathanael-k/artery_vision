#ifndef ARTERYNET_ARTERY_NET_H
#define ARTERYNET_ARTERY_NET_H

#define MAX_DEGREE 10
#define DEFAULT_RADIUS 0.25
#include <vector>

#include <Eigen/Dense>

class arteryGraph;

class arteryNode{
public:
    arteryGraph& graph;
    Eigen::Vector3d position;
    int degree = 0;
    const float radius;
    bool optimized = false;

    arteryNode* junctions[MAX_DEGREE] = {nullptr};
    arteryNode* paths[MAX_DEGREE] = {nullptr};
    float radii[MAX_DEGREE];

    // construct a new arteryNode
    arteryNode(arteryGraph& _graph, const Eigen::Vector3d& posit, float radius = DEFAULT_RADIUS);

    // adds a node to the graph, connected to this node
    arteryNode* addNode(const Eigen::Vector3d& position, float radius = DEFAULT_RADIUS);
};

class arteryGraph{
public:
    arteryNode* root = {nullptr};
    arteryNode* catheter_location = {nullptr};
    std::vector<arteryNode*> junctions;

    // start a new net with some position as first node

    arteryGraph(const Eigen::Vector3d& posit, float radius = DEFAULT_RADIUS);

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

#endif //ARTERYNET_ARTERY_NET_H
