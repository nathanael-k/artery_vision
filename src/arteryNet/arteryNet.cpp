#include <arteryNet.h>

#include <assert.h>

void arteryGraph::connectNodes(arteryNode *node_a, arteryNode *node_b) {

    assert( node_a->degree < MAX_DEGREE &&
            node_b->degree < MAX_DEGREE);

    // update the old node
    node_a->paths[node_a->degree] = node_a;
    node_a->degree++;

    // update the new node
    node_b->paths[node_b->degree] = node_b;
    node_b->degree++;
}

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

    float min_radius = current->radius;

    while (current->degree == 2) {
        // make sure we never track back
        if (current->paths[forward_direction]->degree ==2)
            assert(current->paths[forward_direction]->paths[forward_direction] != current);

        last = current;
        current = current->paths[forward_direction];

        min_radius = std::min(min_radius, current->radius);
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
