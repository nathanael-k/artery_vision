//
// Created by nate on 19.09.20.
//

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc.hpp"

#include <imageData2.h>

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
    initConv.resize(size);
    threshold.resize(size);
    buffer.resize(size);
    components.resize(size);
    for (int i = 0; i < size; i++) {
        source[i].copyTo(visualisation[i]);
        source[i].copyTo(endpoints[i]);
        source[i].copyTo(initConv[i]);
        source[i].copyTo(threshold[i]);
        source[i].copyTo(buffer[i]);
        source[i].copyTo(components[i]);
        //source[i].convertTo(components[i],CV_16UC1);
    }
}

void imageData::resetVisual(from where) {
    if (where == from::source) {
        curr_displayed = &source;
    }
    if (where == from::threshold) {
        if (!threshold[visual_frame].empty()) {
            curr_displayed = &threshold;
        }
    }
    if (where == from::initConv) {
        if (!initConv[visual_frame].empty()) {
            curr_displayed = &initConv;
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
    assert(index < visualisation.size());
    cv::Point A(line[0], line[1]);
    cv::Point B(line[2], line[3]);
    cv::line(visualisation[index], A, B, CV_RGB(100,100,100));
}

void imageData::renderLine(const Eigen::Vector3d& begin, const Eigen::Vector3d& end, int index) {
    assert(index < visualisation.size());
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





Vector2d double_locate(cv::Mat &prio, cv::Mat &backup, const cv::Mat &components, const Eigen::Vector4d &line, const Vector2d pos,
                       const Camera& cam_ref, const Vector2d& pixel_ref, const Camera& cam_new) {

    Vector2d res = locate(prio, components, line, pos, cam_ref, pixel_ref, cam_new);
    if (res == Vector2d(-1,-1)) {
        res = locate(backup, components, line,pos, cam_ref, pixel_ref, cam_new);
    }
    assert(res != Vector2d(-1,-1));
    return res;
}

Vector2d locate(cv::Mat &ref, const cv::Mat &components, const Eigen::Vector4d &line, const Vector2d pos,
                const Camera& cam_ref, const Vector2d& pixel_ref, const Camera& cam_new) {

    struct candidate{Vector2d location; int component; double distance;};
    std::vector<candidate> candidates;
    int weak_index;
    Vector3d dummy;

    // test multiple lines
    // for

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

    Vector2d side = delta.normalized();


    for (int off = -3; off<=3; off++) {

        Vector2d curr = start;
        curr.y() += off * side.x()*15;
        int curr_weak = curr[weak_index];

        for (int i = 0; i < ref.cols; i++) {
            if (ref.at<uchar>(curr.y(), curr.x()) > 200)
                candidates.push_back({curr, components.at<uchar>(curr.y(), curr.x()),
                        Camera::intersect(cam_ref, pixel_ref, cam_new, Vector2d(curr.x(), curr.y()), dummy)});
            ref.at<uchar>(curr.y(), curr.x()) = 99;
            // debug draw to see something
            curr += delta;
            // 4 connected!
            if (curr_weak != curr[weak_index]) {
                Vector2d buff = curr;
                buff[weak_index] = curr_weak;
                if (ref.at<uchar>(buff.y(), buff.x()) > 200)
                    candidates.push_back({curr, components.at<uchar>(buff.y(), buff.x()),
                            Camera::intersect(cam_ref, pixel_ref, cam_new, Vector2d(buff.x(), buff.y()), dummy)});
                curr_weak = curr[weak_index];
                ref.at<uchar>(buff.y(), buff.x()) = 100;
            }
        }
    }

    Vector2d best;

    if(candidates.size() == 0)
        return Vector2d(-1,-1);

    double distance = std::numeric_limits<double>::max();
    int best_component;
    for (auto& candy : candidates) {
        double dist = (candy.location - pos).norm();
        if (dist < distance) {
            best_component = candy.component;
            distance = dist;
        }
    }

    // on the closest component, find the best match
    distance = std::numeric_limits<double>::max();
    for (auto& candy : candidates) {
        if (candy.component == best_component) {


        if (candy.distance < distance) {
            best = candy.location;
            distance = candy.distance;
        }
        }
    }

    return best;
}

