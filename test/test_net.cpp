//
// Created by nate on 16.09.20.
//

#include "Base.h"

#include <arteryNet.h>



void testAdding() {
    arteryGraph graph({0,0,0});
    arteryNode* curr = graph.root;

    // add 5 nodes
    curr = curr->addNode({1,0,0});
    curr = curr->addNode({2,0,0});
    curr = curr->addNode({3,0,0});
    curr = curr->addNode({4,0,0});
    curr = curr->addNode({5,0,0});

    // connect back
    graph.connectNodes(curr,graph.root);

    // we have 6 nodes
    ASSERT_EQUAL( graph.size, 6)

    // no junctions and no endpoints
    ASSERT_EQUAL( graph.nJunction, 0)
    ASSERT_EQUAL( graph.nEnd, 0)

    // add another loop
    // add 5 nodes
    curr = curr->addNode({1,0,1});
    curr = curr->addNode({2,0,1});
    curr = curr->addNode({3,0,1});
    curr = curr->addNode({4,0,1});
    curr = curr->addNode({5,0,1});

    // we have 11 nodes
    ASSERT_EQUAL( graph.size, 11)

    // 2 junctions and 1 endpoints
    ASSERT_EQUAL( graph.nJunction, 1)
    ASSERT_EQUAL( graph.nEnd, 1)

    // connect back
    graph.connectNodes(curr,graph.root);

    // we have 11 nodes
    ASSERT_EQUAL( graph.size, 11)

    // 2 junctions and no endpoints
    ASSERT_EQUAL( graph.nJunction, 2)
    ASSERT_EQUAL( graph.nEnd, 0)

    // add a line with a star at the end
    curr = graph.root->addNode({0,1,0});
    curr = curr->addNode({0,2,0});
    curr = curr->addNode({0,3,0});
    curr->addNode({0,4,0});
    curr->addNode({1,3,0});
    curr->addNode({-1,3,0});
    curr->addNode({0,3,1});
    curr->addNode({0,3,-1});

    // we have 19 nodes
    ASSERT_EQUAL( graph.size, 19)

    // 3 junctions and 5 endpoints
    ASSERT_EQUAL( graph.nJunction, 3)
    ASSERT_EQUAL( graph.nEnd, 5)
}

int main(int, char**)
{
    testAdding();
}

