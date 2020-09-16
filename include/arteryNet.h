#ifndef ARTERYNET_ARTERY_NET_H
#define ARTERYNET_ARTERY_NET_H

#define MAX_DEGREE 10
#include <vector>

#include <Eigen/Dense>

struct arteryNode{
    Eigen::Vector3d position;
    int degree = 0;
    float radius = 0;
    bool optimized = false;

    arteryNode* junctions[MAX_DEGREE] = {nullptr};
    arteryNode* paths[MAX_DEGREE] = {nullptr};
    float radii[MAX_DEGREE];
};

class arteryGraph{

    arteryNode* root = {nullptr};
    arteryNode* catheter_location = {nullptr};
    std::vector<arteryNode*> junctions;

    // establish direct link between junctions for faster processing
    void contractPath(arteryNode* start, int direction);

public:
    // two nodes that should get connected in the graph
    void connectNodes(arteryNode* node_a, arteryNode* node_b);

    // enable direct links between nodes with degree != 2
    void optimize();
    int size = 0;
    bool optimized = false;
};

#endif //ARTERYNET_ARTERY_NET_H
