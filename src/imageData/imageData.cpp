//
// Created by nate on 19.09.20.
//

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc.hpp"

#include <imageData.h>

imageData::imageData(std::string metaFolder, int index) : cam(metaFolder+"meta", index) {
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
        std::string file = metaFolder + cam.name + '_' + buffer + ".png";
        source.push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
    }
    visualisation.resize(size);
    endpoints.resize(size);
    skeleton.resize(size);
    buffer.resize(size);
}

void imageData::resetVisual(from where) {
    if (where == from::source) {
        curr_displayed = &source;
    }
    if (where == from::skeleton) {
        if (!skeleton[visual_frame].empty()) {
            curr_displayed = &skeleton;
        }
    }
    if (where == from::visualisation) {
        if (!visualisation[visual_frame].empty()) {
            curr_displayed = &visualisation;
        }
    }
    if (where == from::endpoints) {
        if (!endpoints[visual_frame].empty()) {
            curr_displayed = &endpoints;
        }
    }
    if (where == from::buffer) {
        if (!buffer[visual_frame].empty()) {
            curr_displayed = &buffer;
        }
    }
}

void imageData::renderLine(Eigen::Vector4d line, int index) {
    cv::Point A(line[0], line[1]);
    cv::Point B(line[2], line[3]);
    cv::line(visualisation[index], A, B, CV_RGB(100,100,100));
}

void imageData::renderPoint(Vector2d point, int index) {
    cv::Point P(point[0], point[1]);
    cv::circle(visualisation[index], P, 3, CV_RGB(100, 100, 255), 2);
}

void imageData::renderPoint(Vector3d point, int index) {
    renderPoint(cam.projectPoint(point), index);
}

void imageData::Skeletonize(int index, bool smooth, bool b_threshold,
                            bool dilate, bool b_thin, int threshold, int max_threshold,
                            int dilation_size) {

    source[index].copyTo(skeleton[index]);

    // blur
    if (smooth) {
        cv::GaussianBlur(skeleton[index], skeleton[index], cv::Size(11,11), 0);
    }


    if (b_threshold) {
        cv::threshold(skeleton[index], skeleton[index], threshold, max_threshold, cv::ThresholdTypes::THRESH_BINARY);
    }

    if (dilate) {
        cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));

        cv::dilate(skeleton[index], skeleton[index], element);
    }

    if (b_thin) {
        //thin(target, smooth, acute_angle, destair); //about 2fps at 1k resolution
        cv::bitwise_not(skeleton[index], skeleton[index]);
        cv::ximgproc::thinning(skeleton[index], skeleton[index], cv::ximgproc::THINNING_GUOHALL); //about 6 fps at 1k resolution
        cv::bitwise_not(skeleton[index], skeleton[index]);
    }
}



void imageData::Endpoints(int index) {
    assert(!skeleton[index].empty());

    endpoints[index] = cv::Mat::zeros(skeleton[index].cols, skeleton[index].rows, skeleton[index].type());
    buffer[index] = cv::Mat::zeros(skeleton[index].cols, skeleton[index].rows, skeleton[index].type());

    // endpoints
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            cv::Mat kernel = (cv::Mat_<int>(3, 3) <<  1, 1, 1, 1,-1, 1, 1, 1, 1);
            kernel.at<int>(i,j)= -1;
            cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, kernel);
            endpoints[index] = endpoints[index] | buffer[index];
        }
    }

    // prepare Kernels
    std::vector<cv::Mat> Kernels(16);
    // "arrows"
    Kernels[0]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,-1,1,    1,1,-1);
    Kernels[1]  = (cv::Mat_<int>(3,3) << 1,-1,1,  1,-1,-1,    -1,1,1);
    Kernels[2]  = (cv::Mat_<int>(3,3) << -1,1,1,  1,-1,-1,    1,-1,1);
    Kernels[3]  = (cv::Mat_<int>(3,3) << 1,1,-1,  -1,-1,1,    1,-1,1);
    // "crosses"
    Kernels[4]  = (cv::Mat_<int>(3,3) << 1,1,1,  -1,-1,-1,    1,-1,1);
    Kernels[5]  = (cv::Mat_<int>(3,3) << 1,-1,1,  1,-1,-1,    1,-1,1);
    Kernels[6]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,-1,1,    1,-1,1);
    Kernels[7]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,-1,-1,    1,1,1);
    // "corners"
    Kernels[8]  = (cv::Mat_<int>(3,3) << 1,1,-1,  1,-1,1,    -1,1,-1);
    Kernels[9]  = (cv::Mat_<int>(3,3) << -1,1,1,  1,-1,1,    -1,1,-1);
    Kernels[10] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,-1,1,    1,1,-1);
    Kernels[11] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,-1,1,    -1,1,1);
    // "rooks"
    Kernels[12] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,-1,1,    1,-1,1);
    Kernels[13] = (cv::Mat_<int>(3,3) << -1,1,1,  1,-1,-1,    -1,1,1);
    Kernels[14] = (cv::Mat_<int>(3,3) << 1,-1,1,  1,-1,1,    -1,1,-1);
    Kernels[15] = (cv::Mat_<int>(3,3) << 1,1,-1,  -1,-1,1,    1,1,-1);

    // splits
    for (int i = 0; i<16; i++) {
            cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, Kernels[i]);
            endpoints[index] = endpoints[index] | (buffer[index] / 2);
    }
}


int correlate(const imageData &lead, const imageData &reference, const Vector2d &leadPixel, const Vector2d &refPixel,
              Vector2d &bestPixel, Vector3d &point, double &distance, int range, int index) {

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

void trace(imageData *lead, imageData *reference, Vector2d leadPixel, Vector2d refPixel, arteryGraph &graph,
           arteryNode *node, std::vector<candidate> &candidates, int index) {
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

void exploreOne(std::vector<candidate> &candidates, arteryGraph &graph) {
    candidate candy = candidates.back();
    candidates.pop_back();
    arteryNode *node = candy.node.addNode(candy.position);
    candy.lead.skeleton[candy.index].at<uchar>(candy.leadPixel.y(), candy.leadPixel.x()) = 100;
    candy.reference.skeleton[candy.index].at<uchar>(candy.refPixel.y(), candy.refPixel.x()) = 100;

    trace(&candy.lead, &candy.reference, candy.leadPixel, candy.refPixel, graph, node, candidates);
}

