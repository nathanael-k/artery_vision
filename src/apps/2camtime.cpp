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

imageData data1("../data/renders/flow_1/", 0),
          data2("../data/renders/flow_1/", 1);

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
    cv::createTrackbar( "Visualisation Source:", "Control", &what, 4, changeVisual);

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
    path,
    endpoint,
    junction
};

struct specialPoint {
    int index;
    pType type = pType::path;
    int processedNeighbours;
    Vector2d posA;
    Vector2d posB;
    double distance;
    int traceLevel = -1;
    bool foundA = false;
    bool foundB = false;
    arteryNode* node = nullptr;
    bool addedGraph = false;
};

bool trace(imageData& data1, imageData& data2, int index, specialPoint& point,
          std::vector<specialPoint>& specialPoints);

int pointCloseTo(const cv::Mat& img, uchar value, Vector2d& position);

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

    arteryGraph graph;
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

        // greedily match endpoints
        // while loop allows us to directly remove found matches

        auto it1 = ends1.begin();
        while(it1 != ends1.end()) {
            auto it2 = ends2.begin();
            while(it2 != ends2.end()) {
                Vector3d position;
                double distance = Camera::intersect(data1.cam.origin, data1.cam.ray(*it1).normalized(),
                                                    data2.cam.origin, data2.cam.ray(*it2).normalized(), position);
                if (distance < 0.25) {
                    // we have a match, create node
                    arteryNode* node = new arteryNode(graph, position);
                    // add to list of Points
                    specialPoint point;
                    point.foundA = point.foundB = true;
                    point.node = node;
                    point.posA = *it1;
                    point.posB = *it2;
                    point.index = locatedPoints.size();
                    node->index = locatedPoints.size();
                    point.distance = distance;
                    locatedPoints.push_back(point);
                    //remove already matched
                    it1 = ends1.erase(it1);
                    it2 = ends2.erase(it2);
                }
                else {
                    it2++;
                }
            }
            it1++;
        }

        // 3) also create nodes for (and collect) unmatched points
        for (auto& end : ends1)
        {
            specialPoint point;
            point.foundA = true;
            point.posA = end;
            locatedPoints.push_back(point);
        }
        for (auto& end : ends2)
        {
            specialPoint point;
            point.foundB = true;
            point.posB = end;
            locatedPoints.push_back(point);
        }

        // 4) mark all points in the current buffer to find them fast
        for (int i = finishedPoints; i < locatedPoints.size(); i++) {
            auto& point = locatedPoints[i];
            if (point.foundA)
                data1.buffer[index].at<uchar>(point.posA.y(), point.posA.x()) = i;
            if (point.foundB)
                data2.buffer[index].at<uchar>(point.posB.y(), point.posB.x()) = i;
        }

        // right now we need at least one matched node, this will be the root node
        assert(locatedPoints[0].foundA && locatedPoints[0].foundB);
        graph.root = locatedPoints[0].node;
        graph.root->enddraw = false;
        finishedPoints++;

        // 5) trace image (on skeleton) beginning at matched endpoints, until you find a marked point in the buffer
        // 6) if that point is already matched, connect in the graph and continue
        // 7) unmatched points can be easily matched because you know locally where you are (thanks to tracing) -> match them and add them to the graph
        for (int i = finishedPoints; i < locatedPoints.size(); i++) {
            auto& point = locatedPoints[i];
            bool success = trace(data1, data2, index, point, locatedPoints);
            assert (success);
        }

        // draw graph into visualisation to check it
        data1.source[index].copyTo(data1.visualisation[index]);
        data2.source[index].copyTo(data2.visualisation[index]);
        data1.drawGraph(*graph.root, index);
        data2.drawGraph(*graph.root, index);


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
                // we have a match
                point.type = pType::endpoint;
            }
            // either way update positions (if already matched to endpoints, will not change here!
            pointCloseTo(data1.skeleton[index+1], 0, point.posA);
            pointCloseTo(data2.skeleton[index+1], 0, point.posB);
            // update node position
            Camera::intersect(data1.cam, point.posA, data2.cam, point.posB, point.node->position);
            // remove on endpoints layer
            data1.endpoints[index+1].at<uchar>(point.posA.y(), point.posA.x()) = 0;
            data2.endpoints[index+1].at<uchar>(point.posB.y(), point.posB.x()) = 0;
            // index on buffer
            data1.buffer[index+1].at<uchar>(point.posA.y(), point.posA.x()) = i;
            data2.buffer[index+1].at<uchar>(point.posB.y(), point.posB.x()) = i;
        }

        debugWaitShow();

        cv::waitKey(0);

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

    // find point in skeleton that is < 255 for img1 and 2
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

    double best1, best2;
    Vector2d pix2_new, pix1_new;
    Vector3d pos1, pos2;
    correlate(img2, cam1, cam2, candidate1, pix2, pix2_new, pos1, best1, 1);
    correlate(img1, cam2, cam1, candidate2, pix1, pix1_new, pos2, best2, 1);

    std::cout << "Match1: " << best1 << " Match2: " << best2 << std::endl;

    if (best1 < best2) {
        pix1 = candidate1;
        pix2 = pix2_new;
        position = pos1;
        return best1;
    }
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
        point.processedNeighbours++;
        other.processedNeighbours++;

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
        f = from::endpoints;
    if (pos == 3)
        f = from::buffer;
    if (pos == 4)
        f = from::visualisation;

    data1.resetVisual(f);
    data2.resetVisual(f);
    displayVisual(0,0);
}

