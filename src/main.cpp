#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"

#include <iostream>
#include <chrono>

#include <camera.h>
#include <arteryNet.h>
#include <imageData.h>

int dilation_size = 4;
int threshold = 140;
int const max_kernel_size = 21;
int const max_threshold = 255;

bool enable_update  =false;
bool b_thin         =true;
bool b_threshold    =true;
bool dilate         =true;
bool smooth         =true;
bool acute_angle    =true;
bool destair        =true;
bool launchTrace    =false;

enum class button {
    dilate,
    smooth,
    acute_angle,
    destair,
    thin,
    threshold,
    launchTrace
};

imageData data1("../data/multi_cam/scene02/", 0),
          data2("../data/multi_cam/scene02/", 1);

void Skeletonize( int, void* );

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void callBackBtn (int i, void* a) {
    button* where{ static_cast<button*>(a)};
    if (*where == button::dilate)           dilate = i;
    if (*where == button::smooth)           smooth = i;
    if (*where == button::acute_angle)      acute_angle = i;
    if (*where == button::destair)          destair = i;
    if (*where == button::thin)             b_thin = i;
    if (*where == button::threshold)        b_threshold = i;
    if (*where == button::launchTrace)      launchTrace = i;
    if (enable_update) Skeletonize(0,0);
};

static void onMouse1(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    data1.pixel = Vector2d(x,y);
    data1.ray = data1.cam.ray(Vector2d(x,y)).normalized();
    Eigen::Vector4d line = data2.cam.projectLine(data1.cam.origin, data1.ray);
    data2.renderLine(line);

    data1.pointRdy = true;
    data1.executeRdy = true;

    Vector3d point;
    if (data2.pointRdy) {
        double distance = Camera::intersect(data1.cam.origin, data1.ray, data2.cam.origin, data2.ray, point);

        if (distance < 0.1) {
            data1.renderPoint(point);
            data2.renderPoint(point);
            data1.pointRdy = false;
            data2.pointRdy = false;
        }

        std::string text = cv::format("Intersection with distance %f.", distance);
        cv::displayStatusBar("Cam1 Source", text);
        cv::displayStatusBar("Cam2 Source", text);
    }

    imshow("Cam1 Source", data1.visualisation[0]);
    imshow("Cam2 Source", data2.visualisation[0]);
}

static void onMouse2(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    data2.pixel = Vector2d(x,y);
    data2.ray = data2.cam.ray(Vector2d(x,y)).normalized();
    Eigen::Vector4d line = data1.cam.projectLine(data2.cam.origin, data2.ray);
    data1.renderLine(line);

    data2.pointRdy = true;
    data2.executeRdy = true;

    Vector3d point;
    if (data1.pointRdy) {
        double distance = Camera::intersect(data1.cam.origin, data1.ray, data2.cam.origin, data2.ray, point);
        if (distance < 0.1) {
            data1.renderPoint(point);
            data2.renderPoint(point);
            data1.pointRdy = false;
            data2.pointRdy = false;
        }

        std::string text = cv::format("Intersection with distance %f.", distance);
        cv::displayStatusBar("Cam1 Source", text);
        cv::displayStatusBar("Cam2 Source", text);
    }

    imshow("Cam1 Source", data1.visualisation[0]);
    imshow("Cam2 Source", data2.visualisation[0]);
}

static void startTrace(int i, void* a) {
    int index = 0;

    // only start if we are ready!
    if (data1.executeRdy == false || data2.executeRdy == false) return;

    // init:
    // find first actual points on skeleton
    bool foundLead = false;
    Vector2d startLead;

    for (int k = 0; k<10; k++) {
        for (int i = -k; i <= k; i++) {
            if (i == k || i == -k) {
            for (int j = -k; j <= k; j++) {
                Vector2d location = data1.pixel + Vector2d(i, j);
                if (data1.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                    startLead = location;
                    foundLead = true;
                    goto endLead;
                }
            }
            }
            else {
                Vector2d location = data1.pixel + Vector2d(i, -k);
                if (data1.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                    startLead = location;
                    foundLead = true;
                    goto endLead;
                }
                location = data1.pixel + Vector2d(i, +k);
                if (data1.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                    startLead = location;
                    foundLead = true;
                    goto endLead;
                }
            }

        }
    }
    endLead:

    bool foundRef = false;
    Vector2d startRef;
    for (int k = 0; k<10; k++) {
        for (int i = -k; i <= k; i++) {
            if (i == k || i == -k) {
                for (int j = -k; j <= k; j++) {
                    Vector2d location = data2.pixel + Vector2d(i, j);
                    if (data2.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                        startRef = location;
                        foundRef = true;
                        goto endRef;
                    }
                }
            }
            else {
                Vector2d location = data2.pixel + Vector2d(i, -k);
                if (data2.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                    startRef = location;
                    foundRef = true;
                    goto endRef;
                }
                location = data2.pixel + Vector2d(i, +k);
                if (data2.skeleton[index].at<uchar>(location.y(), location.x()) == 0) {
                    startRef = location;
                    foundRef = true;
                    goto endRef;
                }
            }

        }
    }
    endRef:

    imshow("Skeleton 1", data1.skeleton[index]);
    imshow("Skeleton 2", data2.skeleton[index]);

    if (!foundLead || !foundRef)
        return;
    assert (foundLead && foundRef);

    Vector3d position;
    Vector2d pixel;
    double distance;

    //search far for the best match
    if (!correlate(data1,data2,startLead,startRef,pixel,position, distance, 10))
        assert (true);

    arteryGraph graph(position);
    std::vector<candidate> candidates;
    candidates.push_back({data1, data2, startLead, pixel, position, *graph.root });
    data1.renderPoint(position);
    data2.renderPoint(position);
    imshow("Cam1 Source", data1.visualisation[0]);
    imshow("Cam2 Source", data2.visualisation[0]);

    int count = 0;

    while (!candidates.empty()) {
        count++;
        data1.renderPoint(candidates.back().position);
        data2.renderPoint(candidates.back().position);
        exploreOne(candidates, graph);
        imshow("Cam1 Source", data1.visualisation[0]);
        imshow("Cam2 Source", data2.visualisation[0]);

        imshow("Skeleton 1", data1.skeleton[0]);
        imshow("Skeleton 2", data2.skeleton[0]);
        //cv::waitKey(1);
    }
    return;
}

int main( int argc, char** argv )
{
    if( data1.source.empty() ||  data2.source.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }

    data1.resetVisual();
    data2.resetVisual();

    // create windows
    cv::namedWindow( "Cam1 Source", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Source", cv::WINDOW_AUTOSIZE);

    cv::namedWindow( "Skeleton 1", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Skeleton 2", cv::WINDOW_AUTOSIZE);

    //cv::namedWindow( "Endpoint Demo", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar( "Threshold:", "Skeleton 1", &threshold, max_threshold, Skeletonize);
    cv::createTrackbar( "Kernel size:\n 2n +1", "Skeleton 1",
                        &dilation_size, max_kernel_size,
                        Skeletonize );
    button choices[7] = {button::dilate, button::smooth, button::acute_angle, button::destair, button::thin, button::threshold, button::launchTrace};
    cv::createButton( "Dilate at beginning", callBackBtn, &choices[0], cv::QT_CHECKBOX, dilate);
    cv::createButton("Thin", callBackBtn, &choices[4], cv::QT_CHECKBOX, b_thin);
    cv::createButton("Threshold", callBackBtn, &choices[5], cv::QT_CHECKBOX, b_threshold);
    cv::createButton("Smooth", callBackBtn, &choices[1], cv::QT_CHECKBOX, smooth);
    cv::createButton("Trace when new origin", startTrace, nullptr, cv::QT_PUSH_BUTTON, 0);

    imshow( "Cam1 Source", data1.source[0]);
    imshow( "Cam2 Source", data2.source[0] );
    enable_update = true;

    // process source images to skeleton
    Skeletonize(0, 0);

    // input starting location to process (will be the position of catheter tip later on)
    // register mouse callback
    cv::setMouseCallback("Cam1 Source", onMouse1);
    cv::setMouseCallback("Cam2 Source", onMouse2);

    // build up graph from that starting point
    cv::waitKey(0);
    return 0;
}


void Skeletonize( int, void* )
{
    auto t1 = std::chrono::high_resolution_clock::now();
    data1.Skeletonize(0, smooth, b_threshold, dilate, b_thin, threshold, max_threshold, dilation_size);
    data2.Skeletonize(0, smooth, b_threshold, dilate, b_thin, threshold, max_threshold, dilation_size);

    imshow( "Skeleton 1", data1.skeleton[0] );
    imshow( "Skeleton 2", data2.skeleton[0] );

    auto t2 = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::string text = cv::format("Time: %dms | FPS: %f", duration, (1000.0/duration));
    cv::displayStatusBar("Skeleton 1", text);
    cv::displayStatusBar("Skeleton 2", text);

    /*

    //Find endpoints: only one pixel in their 8 neighbourhood
    source = target.clone();
    cv::Mat endpoints = cv::Mat::zeros(src.cols, src.rows, src.type());

    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            cv::Mat start = source.clone();
            cv::Mat kernel = (cv::Mat_<int>(3, 3) <<  1, 1, 1, 1,-1, 1, 1, 1, 1);

            kernel.at<int>(i,j)= -1;
            cv::morphologyEx(start, target, cv::MORPH_HITMISS, kernel);
            endpoints = endpoints | target;
        }

    }
    // make endpoints easier to see:
    cv::Mat endpoints_visual = endpoints.clone();
    cv::Mat element = getStructuringElement( 2,
                                             cv::Size( 11, 11 ),
                                             cv::Point( 5, 5 ) );
    cv::morphologyEx(endpoints, endpoints_visual, cv::MORPH_DILATE, element);

    // the following code was a first (unfinished) implementation to track the graph along its lines

    /*

    // build graph
    // find first endpoint, this will be our first node
    // find on top edge / first row
    // opencv is strange: _x denotes the column in this case, unlike when accessing a matrix
    cv::Point start(0,0);
    for (int i = 0; i<endpoints.cols; i++) {
        if (endpoints.at<uchar>(0,i)>0) {
            start.x = i;
            break;
        }
    }
    assert(endpoints.at<uchar>(start) > 0);

    //draw entry point for better visualisation
    endpoints_visual.at<uchar>(start) = 100;

    // unfortunately, we get some artifacts along the border,
    // so lets find the first pixel on the path away from the border
    for (int i = start.x; i < endpoints.cols; i++) {
        if (source.at<uchar>(1, i) == 0) {
            start.x = i;
            start.y = 1;
            break;
        }
        else {
            endpoints_visual.at<uchar>(1, i) = 123;
        }
    }
    assert(start.y == 1);

    endpoints_visual.at<uchar>(start) = 101;


    cv::Point posit = start;
    posit.y++;
    // from that endpoint look in vicinity and continue on the path, creating a path_node every n pixels
    // first just visualize the process!
    for (;posit.y < source.rows; posit.y++) {
        if (source.at<uchar>(posit.y, posit.x-1) == 0)
            posit.x--;
        else if (source.at<uchar>(posit) == 0)
            ;
        else if (source.at<uchar>(posit.y, posit.x+1) == 0)
            posit.x ++;
        else {
            // this should never happen!!!
            // assert(false);
        }
        endpoints_visual.at<uchar> (posit) = 200;
    }

    imshow( "Endpoint Demo", endpoints_visual );
    cv::displayStatusBar("Endpoint Demo", std::to_string(start.y));
    cv::waitKey(0);

    // until you hit something with more than two connections -> a junction -> a new node
    // iteratively proceed on both directions until you are at an endpoint

    // whenever you cant progress, check if it is actually an endpoint as decided earlier
    */
}