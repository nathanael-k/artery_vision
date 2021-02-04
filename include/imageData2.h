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

// enum to define different data to display
enum class from {
    source,
    threshold,
    initConv,
    components,
    visualisation,
    endpoints,
    buffer
};


class imageData {
public:
    int size;
    int visual_frame = 0;

    std::vector<cv::Mat> source, threshold, initConv, components, visualisation, endpoints, buffer;
    std::vector<cv::Mat>* curr_displayed = &source;

    // what is currently selected / set
    Vector2d pixel;
    Vector3d ray;

    bool pointRdy = false;
    bool executeRdy = false;

    arteryGraph graph;

    Camera cam;

    // constructor: loads images from folder based on the meta file
    // prepares all the different image layers by copying from the source
    imageData(std::string metaFolder, int index);

    // #### methods for visualisation

    // starting from node, draw everything connected to it on layer index
    void drawGraph(arteryNode& node, int index = 0) {
        renderPoint(node.position, node.index, index);
        int i = 1;
        if (node.index == 0)
            i = 0;

        for (; i<node.degree; i++) {
            renderLine(node.position, node.paths[i]->position, index);
            drawGraph(*node.paths[i], index);
        }
    }

    // changes the pointer that defines what should currently be displayed
    void resetVisual(from where = from::source);

    // draw a line on "visualization" in image coordinates, at the specified index (image in sequence)
    void renderLine(const Eigen::Vector4d& line, int index = 0);

    // draw a line on "visualization" in 3d coordinates, at the specified index
    void renderLine(const Eigen::Vector3d& begin, const Eigen::Vector3d& end, int index = 0);

    // draw a circle in image coordinates, write label if label is >= 0
    void renderPoint(Vector2d point, int label = -1, int index = 0);

    // draw a circle in 3d coordinates, write label if label is >= 0
    void renderPoint(Vector3d point, int label = -1, int index = 0);

    // #### methods for data preparation

    // prepare all the image layers for the specified index for further processing
    // source:          initial image, does not get touched
    // skeleton:        skeletonized version, black background and white skeleton
    // new_skeleton:    initially a copy of the skeleton, but "old" edges will be suppressed during recognition
    //                  will then show only the skeleton of new parts in the image
    // endpoints:       shows endpoints as 255 and junctions as 127, everything else as 0
    // components:      during processing all edges will become numbered
    // buffer:          ??? used behind the scenes sometimes
    // visualisation:   initially a copy of the source, but we draw onto it as soon as we have progress
    void prepareLayers(int index = 0, bool smooth = true, bool b_threshold = true, bool dilate = true,
                       bool b_thin = true, int threshold = 140, int max_threshold = 255, int dilation_size = 4);

    // call prepareLayers for all images in the sequence
    void prepareAllLayers() {
        for (int i = 0; i<size; i++) {
            prepareLayers(i);
        }
    }

    // makes sure that our endpoint buffer is as required (paint it) at index
    void prepareEndpoints(int index = 0);

    // write everything connected to that node to a file for further processing
    void write_to_file(arteryNode& node, std::ofstream& handle) {
        handle << node.position.x() << " " << node.position.y() << " " << node.position.z() << '\n';
        int i = 1;
        if (node.index == 0)
            i = 0;

        for (; i<node.degree; i++) {
            write_to_file(*node.paths[i], handle);
        }
    }

    // for convenience
    void write_to_file(std::string& file_name) {
        std::ofstream file;
        file.open( "graph.txt" );
        write_to_file(*graph.root, file);
    }

    void apply_threshold(int frame, double thresh, double maxval) {
        cv::threshold(source[frame], threshold[frame], thresh, maxval, cv::THRESH_BINARY);
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


Vector2d double_locate(cv::Mat &prio, cv::Mat &backup, const cv::Mat& components, const Eigen::Vector4d &line, const Vector2d pos, const Camera& cam_ref, const Vector2d& pixel_ref, const Camera& cam_new);

// finds a missing point in reference, by checking the line coordinates for components
// if there are multiple matches... take the closest one?
Vector2d locate(cv::Mat& ref, const cv::Mat& components, const Eigen::Vector4d& line, const Vector2d pos, const Camera& cam_ref, const Vector2d& pixel_ref, const Camera& cam_new);

#endif //ARTERY_VISION_IMAGEDATA_H
