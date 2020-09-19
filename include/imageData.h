//
// Created by nate on 19.09.20.
//
#ifndef ARTERY_VISION_IMAGEDATA_H
#define ARTERY_VISION_IMAGEDATA_H

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>

#include <camera.h>

using Eigen::Vector3d;
using Eigen::Vector2d;

class imageData {
public:
    int size;

    std::vector<cv::Mat> source, skeleton, visualisation, buffer;

    Vector2d pixel;
    Vector3d ray;

    bool pointRdy = false;
    bool executeRdy = false;

    Camera cam;

    imageData(std::string metaFolder, int index) : cam(metaFolder+"meta", index) {
        // read size from file
        std::ifstream inFile;
        inFile.open(metaFolder+"meta");
        if (!inFile) {
            std::cerr << "Unable to open file: " << metaFolder;
            exit(1);   // call system to stop
        }
        // skip 1 line
        std::string textBuffer;
        std::getline(inFile,textBuffer);
        // getSize
        inFile >> size;
        //read all files
        for (int i = 0; i<size; i++) {
            char buffer[50];
            int n = sprintf(buffer, "%03d", i);
            assert(n < 50);
            std::string file = metaFolder + cam.name + buffer + ".png";
            source.push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
        }
        skeleton.resize(size);
        visualisation.resize(size);
        buffer.resize(size);
    }

    void resetVisual(int index = 0) {
        source[index].copyTo(visualisation[index]);
    }

    void renderLine(Eigen::Vector4d line, int index = 0) {
        cv::Point A(line[0], line[1]);
        cv::Point B(line[2], line[3]);
        cv::line(visualisation[index], A, B, CV_RGB(100,100,100));
    }

    void renderPoint(Vector2d point, int index = 0) {
        cv::Point P(point[0], point[1]);
        cv::circle(visualisation[index], P, 3, CV_RGB(100, 100, 255), 2);
    }

    void renderPoint(Vector3d point, int index = 0) {
        renderPoint(cam.projectPoint(point), index);
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
int correlate(const imageData& lead, const imageData& reference,
              const Vector2d& leadPixel, const Vector2d& refPixel, Vector2d& bestPixel,
              Vector3d& point, double& distance, int range = 2, int index = 0) {

    distance = std::numeric_limits<double>::max();
    int area = range + 1;

    // go trough whole neighbourhood
    for (int i = -range; i <= range; i++) {
        for (int j = -range; j <= range; j++) {
            Vector2d location = refPixel + Vector2d(i,j);
            // is it painted?
            if (reference.skeleton[index].at<uchar>(location.y(), location.x()) < 255) {
                Vector3d test;
                double dist = Camera::intersect(lead.cam.origin, lead.cam.ray(leadPixel).normalized(),
                                                reference.cam.origin, reference.cam.ray(location).normalized(),
                                                test);
                // is it closer?
                if (dist < distance) {
                    bestPixel = location;
                    distance = dist;
                    point = test;
                    // is the new point connected?
                    area = std::max(abs(i), abs(j));
                }

            }

        }
    }
    return area;
}

//traces a line, starting from a pixel - correlating two renders / skeletons
//adds to a graph, appending at the node passed as
//marking traced pixels with 100, spawns new traces if other directions are present
//Pre: both imageData have the skeleton populated with black, g
void trace(imageData* lead, imageData* reference, Vector2d leadPixel, Vector2d refPixel,
           arteryGraph& graph, arteryNode* node, std::vector<candidate>& candidates, int index = 0){
    // make sure we are not doing bullshit
    assert(     lead->skeleton[index].at<uchar>(leadPixel.y(), leadPixel.x()) < 255);
    assert(reference->skeleton[index].at<uchar>( refPixel.y(),  refPixel.x()) < 255);

    // add the current pixel, correlated
    Vector3d position;
    Vector2d bestPixel;

    int added = 0;

    // add new candidates from lead pixel
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            Vector2d location = leadPixel + Vector2d(i,j);
            // unchecked pixel?
            if (lead->skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                double distance;
                int radius = correlate(*lead, *reference, leadPixel, refPixel, bestPixel, position, distance);
                if(radius < 2 && distance < 0.02) {
                    candidates.push_back(candidate{*lead, *reference, location, bestPixel, position, *node});
                    lead->skeleton[index].at<uchar>(location.y(), location.x()) = 80;
                    if (reference->skeleton[index].at<uchar>(bestPixel.y(), bestPixel.x()) < 100)
                        reference->skeleton[index].at<uchar>(bestPixel.y(), bestPixel.x()) = 81;
                    added++;
                }
            }
        }
    }

    // add new candidates from reference pixel
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            Vector2d location = refPixel + Vector2d(i,j);
            // unchecked pixel?
            if (reference->skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                double distance;
                int radius = correlate(*reference, *lead, refPixel, leadPixel, bestPixel, position, distance);
                if(radius < 2 && distance < 0.02) {
                    candidates.push_back(candidate{*reference, *lead, location, bestPixel, position, *node});
                    reference->skeleton[index].at<uchar>(location.y(), location.x()) = 82;
                    if (lead->skeleton[index].at<uchar>(bestPixel.y(), bestPixel.x()) < 100)
                        lead->skeleton[index].at<uchar>(bestPixel.y(), bestPixel.x()) = 83;
                    added++;
                }
            }
        }
    }

    if (added == 0) {
        // we added no new candidates, so either we are at an end, or we are closing a loop

    }
}

void exploreOne(std::vector<candidate>& candidates, arteryGraph& graph) {
    candidate candy = candidates.back();
    candidates.pop_back();
    arteryNode *node = candy.node.addNode(candy.position);
    candy.lead.skeleton[candy.index].at<uchar>(candy.leadPixel.y(), candy.leadPixel.x()) = 100;
    candy.reference.skeleton[candy.index].at<uchar>(candy.refPixel.y(), candy.refPixel.x()) = 100;

    trace(&candy.lead, &candy.reference, candy.leadPixel, candy.refPixel, graph, node, candidates);
}


void Skeletonize( imageData& data, bool smooth, bool b_threshold, bool dilate,
                  bool b_thin, int threshold, int max_threshold, int dilation_size, int index = 0) {

    data.source[index].copyTo(data.skeleton[index]);

    // blur
    if (smooth) {
        cv::GaussianBlur(data.skeleton[index], data.skeleton[index], cv::Size(11,11), 0);
    }


    if (b_threshold) {
        cv::threshold(data.skeleton[index], data.skeleton[index], threshold, max_threshold, cv::ThresholdTypes::THRESH_BINARY);
    }

    if (dilate) {
        cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));

        cv::dilate(data.skeleton[index], data.skeleton[index], element);
    }

    if (b_thin) {
        //thin(target, smooth, acute_angle, destair); //about 2fps at 1k resolution
        cv::bitwise_not(data.skeleton[index], data.skeleton[index]);
        cv::ximgproc::thinning(data.skeleton[index], data.skeleton[index], cv::ximgproc::THINNING_GUOHALL); //about 6 fps at 1k resolution
        cv::bitwise_not(data.skeleton[index], data.skeleton[index]);
    }
}

#endif //ARTERY_VISION_IMAGEDATA_H
