//
// Created by nate on 19.09.20.
//
#ifndef ARTERY_VISION_IMAGEDATA_H
#define ARTERY_VISION_IMAGEDATA_H

#include "opencv2/core/utility.hpp"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>

#include <camera.h>
#include <arteryNet.h>

using Eigen::Vector3d;
using Eigen::Vector2d;

enum class from {
    source,
    skeleton,
    visualisation,
    endpoints,
    buffer
};

class imageData {
public:
    int size;
    int visual_frame = 0;

    std::vector<cv::Mat> source, skeleton, visualisation, endpoints, buffer;
    std::vector<cv::Mat>* curr_displayed = &source;

    // what is currently selected / set
    Vector2d pixel;
    Vector3d ray;

    bool pointRdy = false;
    bool executeRdy = false;

    Camera cam;

    imageData(std::string metaFolder, int index);

    void resetVisual(from where = from::source);

    void renderLine(const Eigen::Vector4d& line, int index = 0);

    void renderLine(const Eigen::Vector3d& begin, const Eigen::Vector3d& end, int index = 0);

    void renderPoint(Vector2d point, int index = 0);

    void renderPoint(Vector3d point, int index = 0);

    void Skeletonize(int index = 0, bool smooth = true, bool b_threshold = true, bool dilate = true,
                      bool b_thin = true, int threshold = 140, int max_threshold = 255, int dilation_size = 4);

    void SkeletonizeAll() {
        for (int i = 0; i<size; i++) {
            Skeletonize(i);
        }
    }

    void Endpoints(int index = 0);

    void EndpointsAll() {
        for (int i = 0; i<size; i++) {
            Endpoints(i);
        }
    }

    void drawGraph(arteryNode& node, int index = 0) {
        renderPoint(node.position, index);

        int i = 1;
        if (node.index == 0)
            i = 0;

        for (; i<node.degree; i++) {
            renderLine(node.position, node.paths[i]->position);
            drawGraph(*node.paths[i], index);
        }
    }
};

// a combination that we already know correlates, but we keep it for later
struct candidate {
    imageData& lead;
    imageData& reference;
    Vector2d leadPixel, refPixel;
    Vector3d position;
    arteryNode& node;
    int index = 0;
};

// finds the coordinates where the reference has the best ray trough the lead
// saves bestPixel where the reference picture should be
// distance is how far these points are
// returns pixelDistance: counts how many pixels are between
// -1 means we did not find anything
int correlate(const cv::Mat& ref, const Camera& leadCam, const Camera& refCam,
              const Vector2d& leadPixel, const Vector2d& refPixel, Vector2d& bestPixel,
              Vector3d& point, double& distance, int range = 2, int cutoff = 255);

//traces a line, starting from a pixel - correlating two renders / skeletons
//adds to a graph, appending at the node passed as
//marking traced pixels with 100, spawns new traces if other directions are present
//Pre: both imageData have the skeleton populated with black, g
void trace(imageData* lead, imageData* reference, Vector2d leadPixel, Vector2d refPixel,
           arteryGraph& graph, arteryNode* node, std::vector<candidate>& candidates, int index = 0);

void exploreOne(std::vector<candidate>& candidates, arteryGraph& graph);



#endif //ARTERY_VISION_IMAGEDATA_H
