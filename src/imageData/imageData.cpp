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
    components.resize(size);
    for (int i = 0; i < size; i++) {
        source[i].copyTo(visualisation[i]);
        source[i].copyTo(endpoints[i]);
        source[i].copyTo(skeleton[i]);
        source[i].copyTo(buffer[i]);
        source[i].copyTo(components[i]);
        //source[i].convertTo(components[i],CV_16UC1);
    }
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
    if (where == from::components) {
        if (!components[visual_frame].empty()) {
            curr_displayed = &components;
        }
    }
}

void imageData::renderLine(const Eigen::Vector4d& line, int index) {
    cv::Point A(line[0], line[1]);
    cv::Point B(line[2], line[3]);
    cv::line(visualisation[index], A, B, CV_RGB(100,100,100));
}


void imageData::renderLine(const Eigen::Vector3d& begin, const Eigen::Vector3d& end, int index) {
    Vector2d a,b;
    a = cam.projectPoint(begin); b = cam.projectPoint(end);
    cv::Point A(a[0],a[1]);
    cv::Point B(b[0],b[1]);
    cv::line(visualisation[index], A, B, CV_RGB(255,100,100));
}



void imageData::renderPoint(Vector2d point, int label, int index) {
    cv::Point P(point[0], point[1]);
    cv::circle(visualisation[index], P, 3, CV_RGB(100, 100, 255), 2);
    if (label >= 0)
        cv::putText(visualisation[index], std::to_string(label), P, 0, 1, CV_RGB(255,255,255));
}

void imageData::renderPoint(Vector3d point, int label, int index) {
    renderPoint(cam.projectPoint(point), label, index);
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
        //cv::bitwise_not(skeleton[index], skeleton[index]);
    }

    // remove small stubs ( https://en.wikipedia.org/wiki/Pruning_(morphology) )
    std::vector<cv::Mat> ThinningKernels(8);
    ThinningKernels[0] = (cv::Mat_<int>(3,3) << 0,-1,-1,  1,1,-1,    0,-1,-1);
    ThinningKernels[1] = (cv::Mat_<int>(3,3) << 0,1,0,  -1,1,-1,    -1,-1,-1);
    ThinningKernels[2] = (cv::Mat_<int>(3,3) << -1,-1,0,  -1,1,1,    -1,-1,0);
    ThinningKernels[3] = (cv::Mat_<int>(3,3) << -1,-1,-1,  -1,1,-1,    0,1,0);

    ThinningKernels[4] = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,-1,    -1,-1,-1);
    ThinningKernels[5] = (cv::Mat_<int>(3,3) << -1,-1,1,  -1,1,-1,    -1,-1,-1);
    ThinningKernels[6] = (cv::Mat_<int>(3,3) << -1,-1,-1,  -1,1,-1,    1,-1,-1);
    ThinningKernels[7] = (cv::Mat_<int>(3,3) << -1,-1,-1,  -1,1,-1,    -1,-1,1);

    // thin 10 times
    for (int i = 0; i<10; i++) {
        for (int ker = 0; ker<8; ker++) {
            cv::morphologyEx(skeleton[index],buffer[index], cv::MORPH_HITMISS, ThinningKernels[ker]);
            skeleton[index] = skeleton[index] - (buffer[index]);
        }
    }

    // now we have a skeleton, but we can still have T shapes and other problems, so lets remove some more pixels

    // for T shapes, we just have to remove the pixel at the top of the T in the middle
    // prepare Kernels
    std::vector<cv::Mat> Kernels(4);
    // "T"
    Kernels[0] = (cv::Mat_<int>(3,3) << -1,-1,-1,  1,1,1,    -1,1,-1);
    Kernels[1] = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,1,    -1,1,-1);
    Kernels[2] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,-1,    -1,1,-1);
    Kernels[3] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,1,    -1,-1,-1);

    for (int i = 0; i<4; i++) {
        cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, Kernels[i]);
        skeleton[index] = skeleton[index] - (buffer[index]);
    }

    // we can also have arrows:
    // X         x
    //  xx   ->  x x
    //  x         x
    Kernels[0] = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,1,    -1,1,-1);
    Kernels[1] = (cv::Mat_<int>(3,3) << -1,-1,1,  1,1,-1,    -1,1,-1);
    Kernels[2] = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,-1,    -1,-1,1);
    Kernels[3] = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,1,    1,-1,-1);

    std::vector<cv::Point> Points(4);
    Points[0] = cv::Point(1,0);
    Points[1] = cv::Point(1,0);
    Points[2] = cv::Point(2,1);
    Points[3] = cv::Point(1,2);


    endpoints[index] = cv::Mat::zeros(skeleton[index].cols, skeleton[index].rows, skeleton[index].type());
    // cache points to add
    for (int i = 0; i<4; i++) {
        cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, Kernels[i], Points[i]);
        endpoints[index] = endpoints[index] + (buffer[index]);
    }
    // remove points
    for (int i = 0; i<4; i++) {
        cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, Kernels[i]);
        skeleton[index] = skeleton[index] - (buffer[index]);
    }

    // strange pixels appear on the horizon
    int max = skeleton[index].cols;
    endpoints[index].at<uchar>(0,0) = 0;
    endpoints[index].at<uchar>(max-1,0) = 0;
    endpoints[index].at<uchar>(0,max-1) = 0;
    endpoints[index].at<uchar>(max-1,max-1) = 0;
    // add prepared points
    skeleton[index] = skeleton[index] + endpoints[index];
}



void imageData::Endpoints(int index) {
    assert(!skeleton[index].empty());

    endpoints[index] = cv::Mat::zeros(skeleton[index].cols, skeleton[index].rows, skeleton[index].type());
    buffer[index] = cv::Mat::zeros(skeleton[index].cols, skeleton[index].rows, skeleton[index].type());

    // endpoints
    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            cv::Mat kernel = (cv::Mat_<int>(3, 3) <<  -1, -1, -1, -1,1, -1, -1, -1, -1);
            kernel.at<int>(i,j)= 1;
            cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, kernel);
            endpoints[index] = endpoints[index] | buffer[index];
        }
    }

    // prepare Kernels
    std::vector<cv::Mat> Kernels(8);
    // "corners"
    Kernels[0]  = (cv::Mat_<int>(3,3) << -1,-1,1,  -1,1,-1,    1,-1,1);
    Kernels[1]  = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,-1,    1,-1,1);
    Kernels[2]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    -1,-1,1);
    Kernels[3]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    1,-1,-1);
    // "rooks"
    Kernels[4]  = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    -1,1,-1);
    Kernels[5]  = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,1,    1,-1,-1);
    Kernels[6]  = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,-1,    1,-1,1);
    Kernels[7]  = (cv::Mat_<int>(3,3) << -1,-1,1,  1,1,-1,    -1,-1,1);


    /*
    // "arrows"
    Kernels[0]  = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,-1,    -1,-1,1);
    Kernels[1]  = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,1,    1,-1,-1);
    Kernels[2]  = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,1,    -1,1,-1);
    Kernels[3]  = (cv::Mat_<int>(3,3) << -1,-1,1,  1,1,-1,    -1,1,-1);
    // "crosses"
    Kernels[4]  = (cv::Mat_<int>(3,3) << -1,-1,-1,  1,1,1,    -1,1,-1);
    Kernels[5]  = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,1,    -1,1,-1);
    Kernels[6]  = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,-1,    -1,1,-1);
    Kernels[7]  = (cv::Mat_<int>(3,3) << -1,1,-1,  1,1,1,    -1,-1,-1);
    // "corners"
    Kernels[8]  = (cv::Mat_<int>(3,3) << -1,-1,1,  -1,1,-1,    1,-1,1);
    Kernels[9]  = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,-1,    1,-1,1);
    Kernels[10] = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    -1,-1,1);
    Kernels[11] = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    1,-1,-1);
    // "rooks"
    Kernels[12] = (cv::Mat_<int>(3,3) << 1,-1,1,  -1,1,-1,    -1,1,-1);
    Kernels[13] = (cv::Mat_<int>(3,3) << 1,-1,-1,  -1,1,1,    1,-1,-1);
    Kernels[14] = (cv::Mat_<int>(3,3) << -1,1,-1,  -1,1,-1,    1,-1,1);
    Kernels[15] = (cv::Mat_<int>(3,3) << -1,-1,1,  1,1,-1,    -1,-1,1);
     */

    // splits
    for (int i = 0; i<8; i++) {
            cv::morphologyEx(skeleton[index], buffer[index], cv::MORPH_HITMISS, Kernels[i]);
            endpoints[index] = endpoints[index] | (buffer[index] / 2);
    }
}



int correlate(const cv::Mat& ref, const Camera& leadCam, const Camera& refCam,
              const Vector2d& leadPixel, const Vector2d& refPixel, Vector2d& bestPixel,
              Vector3d& point, double& distance, int range, int cutoff) {

    distance = std::numeric_limits<double>::max();
    int pixelDistance = -1;

    // go trough whole neighbourhood
    for (int i = -range; i <= range; i++) {
        for (int j = -range; j <= range; j++) {
            Vector2d location = refPixel + Vector2d(i,j);
            // is it part of the skeleton?
            if (ref.at<uchar>(location.y(), location.x()) < cutoff) {
                Vector3d test;
                double dist = Camera::intersect(leadCam, leadPixel,
                                                refCam, refPixel,
                                                test);
                // is it closer?
                if (dist < distance) {
                    bestPixel = location;
                    distance = dist;
                    point = test;
                    // is the new point connected?
                    pixelDistance = std::max(abs(i), abs(j));
                }

            }

        }
    }
    return pixelDistance;
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
                int radius = correlate(reference->skeleton[index], lead->cam, reference->cam,
                                       leadPixel, refPixel, bestPixel, position, distance);
                assert(radius != -1);
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
                int radius = correlate(lead->skeleton[index], reference->cam, lead->cam,
                                       refPixel, leadPixel, bestPixel, position, distance);
                assert(radius != -1);
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

Vector2d locate(cv::Mat &ref, const Eigen::Vector4d &line, const Vector2d pos) {

    struct candidate{Vector2d location; int component;};
    std::vector<candidate> candidates;
    int weak_index;

    // we basically draw a line, but make sure that it is 4 connect or we could miss a component!
    Vector2d start(line[0], line[1]);
    Vector2d end(line[2], line[3]);
    Vector2d delta = end - start;
    if (abs(delta.x()) > abs(delta.y())) {
        delta /= abs(delta.x());
        weak_index = 0;
    }
    else{
        delta /= abs(delta.y());
        weak_index = 1;
    }

    Vector2d curr = start;
    int curr_weak = curr[weak_index];

    for (int i = 0; i<ref.cols; i++) {
        if (ref.at<uchar>(curr.y(), curr.x()) > 0)
            candidates.push_back({curr, ref.at<uchar>(curr.y(), curr.x())});
        // debug draw to see something
        curr += delta;
        // 4 connected!
        if (curr_weak != curr[weak_index]) {
            Vector2d buff = curr;
            buff[weak_index] = curr_weak;
            if (ref.at<uchar>(buff.y(), buff.x()) > 0)
                candidates.push_back({curr, ref.at<uchar>(buff.y(), buff.x())});
            curr_weak = curr[weak_index];
        }
    }

    Vector2d best;

    assert(candidates.size() > 0);

    double distance = std::numeric_limits<double>::max();
    for (auto& candy : candidates) {
        double dist = (candy.location - pos).norm();
        if (dist < distance) {
            best = candy.location;
        }
    }

    return best;
}

