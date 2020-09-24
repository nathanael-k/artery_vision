//
// Created by nate on 19.09.20.
//

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"

#include <iostream>
#include <chrono>

#include <camera.h>
#include <arteryNet.h>
#include <imageData.h>
#include <list>

imageData data1("../data/renders/easy_flow_2/", 0),
          data2("../data/renders/easy_flow_2/", 1);

void displayVisual( int, void* );
void changeVisual( int pos, void* );
void buildGraph(imageData& data1, imageData& data2);

void debugWaitShow() {
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
    cv::waitKey(0);
}

int main( int argc, char** argv )
{
    if( data1.source.empty() ||  data2.source.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }


    // create windows
    cv::namedWindow( "Control");
    cv::namedWindow( "Cam1 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Visual", cv::WINDOW_AUTOSIZE);
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);

    int what = 0;

    cv::createTrackbar( "Frame:", "Control", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Visualisation Source:", "Control", &what, 5, changeVisual);

    // prepare skeletons
    data1.SkeletonizeAll();
    data2.SkeletonizeAll();

    // prepare endpoints
    data1.EndpointsAll();
    data2.EndpointsAll();

    buildGraph(data1, data2);

    // input starting location to process (will be the position of catheter tip later on)
    // register mouse callback
    //cv::setMouseCallback("Cam1 Source", onMouse1);
    //cv::setMouseCallback("Cam2 Source", onMouse2);

    cv::waitKey(0);
    return 0;
}

// to classify type of special points
enum class pType {
    endpoint,
    path,
    junction,
    pseudo_junction
};

struct specialPoint {
    int index;
    pType type = pType::endpoint;
    Vector2d posA;
    Vector2d posB;
    double distance;
    bool foundA = false;
    bool foundB = false;
    arteryNode* node = nullptr;
    bool addedGraph = false;
};

bool trace(imageData& data1, imageData& data2, int index, specialPoint& point,
          std::vector<specialPoint>& specialPoints);

int pointCloseTo(const cv::Mat& img, uchar value, Vector2d& position);

struct Edge{
    // indexA < indexB
    int indexA_ = -1;
    int indexB_ = -1;
    bool added_to_graph;

    void add_index(int index) {
        if (indexA_ == -1)
            indexA_ = index;
        else
        {
            assert(indexB_ == -1);
            if (indexA_ < index)
                indexB_ = index;
            else {
                indexB_ = indexA_;
                indexA_ = index;
            }
        }
    }
};

int processEdges(const std::vector<Edge>& edges, std::vector<specialPoint>& points, int index = 0);

void matchPoints(std::list<Vector2d>& list1, std::list<Vector2d>& list2,
                 const Camera& cam1, const Camera& cam2, arteryGraph& graph,
                 std::vector<specialPoint>& points, pType type) {
    // greedily match endpoints
    // while loop allows us to directly remove found matches
    auto it1 = list1.begin();
    while(it1 != list1.end()) {
        auto it2 = list2.begin();
        while(it2 != list2.end()) {
            Vector3d position;
            double distance = Camera::intersect(cam1.origin, cam1.ray(*it1).normalized(),
                                                cam2.origin, cam2.ray(*it2).normalized(), position);
            if (distance < 0.25) {
                // we have a match, create node
                arteryNode* node = new arteryNode(graph, position);
                // add to list of Points
                specialPoint point;
                point.foundA = point.foundB = true;
                point.node = node;
                point.posA = *it1;
                point.posB = *it2;
                point.index = points.size();
                node->index = points.size();
                point.distance = distance;
                point.type = type;
                points.push_back(point);
                //remove already matched
                it1 = list1.erase(it1);
                it2 = list2.erase(it2);
            }
            else {
                it2++;
            }
        }
        it1++;
    }
}

// builds a graph from matching image sequences
// 1) find endpoints or junctions in both frames at same time
// 2) greedily match them to find 3d positions, add those to the graph
// 3) also create nodes for (and collect) unmatched points
// 4) mark all points in the buffer to find them fast
// 5) trace image (on skeleton) beginning at matched endpoints, until you find a marked point in the buffer
// 6) if that point is already matched, connect in the graph and continue
// 7) unmatched points can be easily matched because you know locally where you are (thanks to tracing) -> match them and add them to the graph
// when all are matched, correlate those matches to next endpoints layer / skeleton
// junctions are expected to stay, however if a path appears again its now a (final) endpoint
// for a match: delete the respective pixel in the endpoints layer
// paint all known matches in the buffer layer of the next time step (at correct location)
void buildGraph(imageData& data1, imageData& data2) {

    arteryGraph& graph = data1.graph;
    std::vector<specialPoint> locatedPoints;
    int finishedPoints = 0;

    // prepare
    cv::bitwise_not(data1.endpoints[0], data1.buffer[0]);
    cv::bitwise_not(data2.endpoints[0], data2.buffer[0]);

    // loop over all images in the sequence
    for (int index = 0; index<data1.size; index++) {

        // find special points (endpoints or junctions) in both frames
        std::list<Vector2d> ends1;
        std::list<Vector2d> ends2;
        std::list<Vector2d> junctions1;
        std::list<Vector2d> junctions2;


        // iterate over whole picture, put special pixel into vector to process them later
        for (int j = 0; j < data1.endpoints[index].rows; j++) {

            // for fast matrix access
            uchar* pixel1 = data1.endpoints[index].ptr<uchar>(j);
            uchar* pixel2 = data2.endpoints[index].ptr<uchar>(j);

            for (int k = 0; k<data1.endpoints[index].cols; k++) {
                if (*pixel1 == 255)
                    ends1.emplace_back(Vector2d(k,j));
                if (*pixel1 == 128)
                    junctions1.emplace_back(Vector2d(k,j));
                if (*pixel2 == 255)
                    ends2.emplace_back(Vector2d(k,j));
                if (*pixel2 == 128)
                    junctions2.emplace_back(Vector2d(k,j));
                pixel1++;
                pixel2++;
            }
        }

        matchPoints(junctions1, junctions2, data1.cam, data2.cam,graph,locatedPoints, pType::junction);


        // 3) also create nodes for (and collect) unmatched points
        for (auto& junction : junctions1)
        {
            specialPoint point;
            point.foundA = true;
            point.posA = junction;
            point.index = locatedPoints.size();
            point.type = pType::junction;
            locatedPoints.push_back(point);
        }
        for (auto& junction : junctions2)
        {
            specialPoint point;
            point.foundB = true;
            point.posB = junction;
            point.index = locatedPoints.size();
            point.type = pType::junction;
            locatedPoints.push_back(point);
        }

        matchPoints(ends1, ends2, data1.cam, data2.cam,graph,locatedPoints, pType::endpoint);

        for (auto& end : ends1)
        {
            specialPoint point;
            point.foundA = true;
            point.posA = end;
            point.index = locatedPoints.size();
            locatedPoints.push_back(point);
        }
        for (auto& end : ends2)
        {
            specialPoint point;
            point.foundB = true;
            point.posB = end;
            point.index = locatedPoints.size();
            locatedPoints.push_back(point);
        }


        // 4) mark all points in the current buffer to find them fast
        // also disconnect skeleton at the points so that we can find the components
        // TODO: check if needed!
        for (int i = finishedPoints; i < locatedPoints.size(); i++) {
            auto& point = locatedPoints[i];
            if (point.foundA){
                data1.buffer[index].at<uchar>(point.posA.y(), point.posA.x()) = i;
                data1.skeleton[index].at<uchar>(point.posA.y(), point.posA.x()) = 0;

            }
            if (point.foundB) {
                data2.buffer[index].at<uchar>(point.posB.y(), point.posB.x()) = i;
                data2.skeleton[index].at<uchar>(point.posB.y(), point.posB.x()) = 0;
            }
        }

        // mark/find connected components of skeleton
        // TODO: remove and put to preprocessing (how to handle endge_count?)
        int edge_count_1 = cv::connectedComponents(data1.skeleton[index], data1.components[index], 8, CV_16U) -1;
        int edge_count_2 = cv::connectedComponents(data2.skeleton[index], data2.components[index], 8, CV_16U) -1;
        data1.components[index].convertTo(data1.components[index], CV_8U);
        data2.components[index].convertTo(data2.components[index], CV_8U);

        //if (index == 1)
          //  debugWaitShow();

        // collect edges
        std::vector<Edge> edges_in_1(edge_count_1);
        std::vector<Edge> edges_in_2(edge_count_2);
        for(auto& point : locatedPoints) {
            // find out what edge this point is connected to
            int max_connections = 1;
            if (point.type == pType::path)     max_connections = 2;
            if (point.type == pType::junction) max_connections = 3;
            int found_connections_cam1 = 0;
            int found_connections_cam2 = 0;

            // work on camera 1 first
            if (point.foundA)
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    if (i != 0 || j != 0) {
                        Vector2d trialPoint = point.posA + Vector2d(i, j);
                        int edge = data1.components[index].at<uchar>(trialPoint.y(), trialPoint.x());
                        if (edge != 0) {
                            found_connections_cam1++;
                            edges_in_1[edge-1].add_index(point.index);
                        }
                    }
            // work on camera 2
            if (point.foundB)
                for (int i = -1; i <= 1; i++)
                    for (int j = -1; j <= 1; j++)
                        if (i != 0 || j != 0) {
                            Vector2d trialPoint = point.posB + Vector2d(i, j);
                            int edge = data2.components[index].at<uchar>(trialPoint.y(), trialPoint.x());
                            if (edge != 0) {
                                found_connections_cam2++;
                                edges_in_2[edge-1].add_index(point.index);
                            }
                        }
            // this is only relevant for new points!
            if (!point.addedGraph)
                assert(found_connections_cam1 <= max_connections &&
                    found_connections_cam2 <= max_connections);

        }

        // only first round, add root
        if (index == 0) {
            // right now we need at least one matched node, this will be the root node
            assert(locatedPoints[0].foundA && locatedPoints[0].foundB);
            locatedPoints[0].addedGraph = true;
            graph.root = locatedPoints[0].node;
            finishedPoints = 1;
        }

        int sum = 100;
        while (sum > 0) {
            int edges_1_left = processEdges(edges_in_1, locatedPoints, index);
            int edges_2_left = processEdges(edges_in_2, locatedPoints, index);
            sum = edges_1_left + edges_2_left;
        }

        // at this moment, all Points should have been added to the graph
        for (auto& point : locatedPoints)
            assert(point.addedGraph);

        // draw graph into visualisation to check it
        data1.source[index].copyTo(data1.visualisation[index]);
        data2.source[index].copyTo(data2.visualisation[index]);
        data1.drawGraph(*graph.root, index);
        data2.drawGraph(*graph.root, index);

        //if (index == 1)
            debugWaitShow();

        // when all are matched, correlate those matches to next endpoints layer / skeleton
        // junctions are expected to stay, however if a path appears again its now a (final) endpoint
        // for a match: delete the respective pixel in the endpoints layer
        // paint all known matches in the buffer layer of the next time step (at correct location)
        cv::bitwise_not(data1.endpoints[index+1], data1.buffer[index+1]);
        cv::bitwise_not(data2.endpoints[index+1], data2.buffer[index+1]);
        for (int i = 0; i < locatedPoints.size(); i++) {
            auto& point = locatedPoints[i];
            // try to point close to and update endpoints
            // this happens if we have a "real" endpoint, that will match multiple rounds
            if (pointCloseTo(data1.endpoints[index+1], 255, point.posA) != -1 &&
                pointCloseTo(data2.endpoints[index+1], 255, point.posB) != -1) {
            }
            else {
                point.type = pType::path;
            }
            // either way update positions (if already matched to endpoints, will not change here!)
            pointCloseTo(data1.skeleton[index+1], 255, point.posA);
            pointCloseTo(data2.skeleton[index+1], 255, point.posB);
            // update node position
            Camera::intersect(data1.cam, point.posA, data2.cam, point.posB, point.node->position);
            // remove on endpoints layer
            data1.endpoints[index+1].at<uchar>(point.posA.y(), point.posA.x()) = 0;
            data2.endpoints[index+1].at<uchar>(point.posB.y(), point.posB.x()) = 0;
            // index on buffer
            data1.buffer[index+1].at<uchar>(point.posA.y(), point.posA.x()) = i;
            data2.buffer[index+1].at<uchar>(point.posB.y(), point.posB.x()) = i;
            // remove from next skeleton
            data1.skeleton[index+1].at<uchar>(point.posA.y(), point.posA.x()) = 0;
            data2.skeleton[index+1].at<uchar>(point.posB.y(), point.posB.x()) = 0;

        }
        std::string egal;
        data1.write_to_file(egal );
        //debugWaitShow();

        // trace and delete the whole "old" part of the graph!
        // is it necessary???

        // declare this part as finished!
        finishedPoints = locatedPoints.size();
    }
}

// finds the closest point with specific value close to position. modifies position and returns how many pixels difference we had
int pointCloseTo(const cv::Mat& img, uchar value, Vector2d& position) {

    for (int k = 0; k<10; k++) {
        for (int i = -k; i <= k; i++) {
            if (i == k || i == -k) {
                for (int j = -k; j <= k; j++) {
                    Vector2d location = position + Vector2d(i, j);
                    if (img.at<uchar>(location.y(), location.x()) == value) {
                        position = location;
                        return std::max(std::abs(i),std::abs(j));
                    }
                }
            }
            else {
                Vector2d location = position + Vector2d(i, -k);
                if (img.at<uchar>(location.y(), location.x()) == value) {
                    position = location;
                    return std::max(std::abs(i),std::abs(k));
                }
                location = position + Vector2d(i, +k);
                if (img.at<uchar>(location.y(), location.x()) == value) {
                    position = location;
                    return std::max(std::abs(i),std::abs(k));
                }
            }
        }
    }

    // signal failure
    return -1;
}

// tries to find best match, returns distance
double bestMatch(cv::Mat& img1, Camera& cam1, Vector2d& pix1, cv::Mat& img2, Camera& cam2, Vector2d& pix2, Vector3d& position) {

    // find point in skeleton that is == 255 for img1 and 2
    Vector2d candidate1;
    bool candidate1found = false;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            if (i != 0 || j != 0) {
                Vector2d trialPoint = pix1 + Vector2d(i, j);
                if (img1.at<uchar>(trialPoint.y(), trialPoint.x()) == 0) {
                    candidate1 = trialPoint;
                    assert (!candidate1found);
                    candidate1found = true;
                }
            }

    Vector2d candidate2;
    bool candidate2found = false;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            if (i != 0 || j != 0) {
                Vector2d trialPoint = pix2 + Vector2d(i, j);
                if (img2.at<uchar>(trialPoint.y(), trialPoint.x()) == 0) {
                    candidate2 = trialPoint;
                    assert (!candidate2found);
                    candidate2found = true;
                }
            }

    assert(candidate1found && candidate2found);

    // to which do these candidates fit best?
    double best1, best2;
    Vector2d pix2_new, pix1_new;
    Vector3d pos1, pos2;
    correlate(img2, cam1, cam2, candidate1, pix2, pix2_new, pos1, best1, 1, 2);
    correlate(img1, cam2, cam1, candidate2, pix1, pix1_new, pos2, best2, 1, 2);

    std::cout << "Match1: " << best1 << " Match2: " << best2 << std::endl;

    if (best1 < best2) {
        // never go back if we left!
        img1.at<uchar>(pix1.y(), pix1.x()) = 2;
        pix1 = candidate1;
        pix2 = pix2_new;
        position = pos1;
        return best1;
    }
    img2.at<uchar>(pix2.y(), pix2.x()) = 2;
    pix2 = candidate2;
    pix1 = pix1_new;
    position = pos2;
    return best2;
}

bool trace(imageData& data1, imageData& data2, int index, specialPoint& point,
                  std::vector<specialPoint>& specialPoints) {

    assert (point.foundA && point.foundB && point.node);
    Vector2d locA = point.posA;
    Vector2d locB = point.posB;

nextPixel:
    // mark pixel in skeleton as processed
    data1.skeleton[index].at<uchar>(locA.y(), locA.x()) = 1;
    data2.skeleton[index].at<uchar>(locB.y(), locB.x()) = 1;

    Vector3d position;
    // directly get new positions where we matched
    double distance = bestMatch(data1.skeleton[index], data1.cam, locA,
                                data2.skeleton[index], data2.cam, locB,
                                position);

    // check if pixel is special:
    int otherIndex = -1;

    if (data1.buffer[index].at<uchar>(locA.y(), locA.x()) != 255)
        otherIndex = data1.buffer[index].at<uchar>(locA.y(), locA.x());
    else if (data2.buffer[index].at<uchar>(locB.y(), locB.x()) != 255)
        otherIndex = data2.buffer[index].at<uchar>(locB.y(), locB.x());
    // was not special, continue trace (make sure we dont think our pixel is special)
    if (otherIndex == -1 || otherIndex == point.index)
        goto nextPixel;
    else {
        // it was special!!!
        specialPoint& other = specialPoints[otherIndex];

        if (other.node)
            point.node->graph.connectNodes(point.node, other.node);
        else
            other.node = point.node->addNode(position);

        if (!other.foundA) {
            other.foundA = true;
            other.posA = locA;
        }
        if (!other.foundB) {
            other.foundB = true;
            other.posB = locB;
        }
        data1.skeleton[index].at<uchar>(locA.y(), locA.x()) = 1;
        data2.skeleton[index].at<uchar>(locB.y(), locB.x()) = 1;
        return true;
    }
}

int processEdges(const std::vector<Edge>& edges, std::vector<specialPoint>& points, int index){

    int open_edges = 0;

    // in an ideal world, all edges would now connect two locatedPoints
    for (auto& edge : edges){
        // both sides connected of the edge
        assert(edge.indexA_ != -1 && edge.indexB_ != -1);
        // we only need to do something if one of the nodes is not yet in the graph
        specialPoint& A = points[edge.indexA_];
        specialPoint& B = points[edge.indexB_];
        if (A.addedGraph && B.addedGraph){}
            // do nothing
        else if (A.addedGraph) {
            assert(A.foundA && A.foundB && A.node && !B.addedGraph);

            // we know the point B is not yet added to the graph
            // if it is already found on both cameras, we have good location and can add to the graph
            if (B.foundA && B.foundB) {
                assert(B.node);
                A.node->graph.connectNodes(A.node, B.node);
                B.addedGraph = true;
            }
            // point is only on one side yet
            else {
                if (!B.foundA) {
                    assert(B.foundB);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.cam.ray(B.posB));
                    data1.renderLine(line, index);
                    B.posA = locate(data1.components[index], line, A.posA);
                    B.foundA = true;
                }
                if (!B.foundB) {
                    assert(B.foundA);
                    // for debugging, first really actually draw this line in visualisation
                    Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.cam.ray(B.posA));
                    data2.renderLine(line, index);
                    B.posB = locate(data2.components[index], line, B.posB);
                    B.foundB = true;
                }
                assert(!B.node);
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(B.posA).normalized(),
                                                    data2.cam.origin, data2.cam.ray(B.posB).normalized(), position);
                assert (distance < 30);
                B.node = A.node->addNode(position);
                B.node->index = edge.indexB_;
                B.addedGraph = true;
            }
        }
        else {
            // both special points associated with this edge were not yet added to the graph
            open_edges ++;

        }
    }
    return open_edges;
}

void displayVisual( int, void* )
{
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
}

void changeVisual( int pos, void* )
{
    from f;
    if (pos == 0)
        f = from::source;
    if (pos == 1)
        f = from::skeleton;
    if (pos == 2)
        f = from::components;
    if (pos == 3)
        f = from::endpoints;
    if (pos == 4)
        f = from::buffer;
    if (pos == 5)
        f = from::visualisation;

    data1.resetVisual(f);
    data2.resetVisual(f);
    displayVisual(0,0);
}

