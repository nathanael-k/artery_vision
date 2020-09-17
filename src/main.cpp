#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"

#include "zhangsuen.h"

#include <iostream>
#include <chrono>

#include <camera.h>
#include <arteryNet.h>

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 4;
int threshold = 140;
int const max_elem = 2;
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

class imageData {
public:
    cv::Mat source, skeleton, visualisation, buffer;

    Vector2d pixel;
    Vector3d ray;

    bool pointRdy = false;
    bool executeRdy = false;

    Camera* cam;

    void resetVisual() {
        source.copyTo(visualisation);
    }

    void renderLine(Eigen::Vector4d line) {
        cv::Point A(line[0], line[1]);
        cv::Point B(line[2], line[3]);
        cv::line(visualisation, A, B, CV_RGB(100,100,100));
    }

    void renderPoint(Vector2d point) {
        cv::Point P(point[0], point[1]);
        cv::circle(visualisation, P, 3, CV_RGB(100, 100, 255), 2);
    }

    void renderPoint(Vector3d point) {
        renderPoint(cam->projectPoint(point));
    }
};

// finds the coordinates where the reference has the best ray trough the lead
int correlate(const imageData& lead, const imageData& reference,
               const Vector2d& leadPixel, const Vector2d& refPixel, Vector2d& bestPixel,
               Vector3d& point, double& distance, int range = 2) {

    distance = std::numeric_limits<double>::max();
    int area = range + 1;

    // go trough whole neighbourhood
    for (int i = -range; i <= range; i++) {
        for (int j = -range; j <= range; j++) {
            Vector2d location = refPixel + Vector2d(i,j);
            // is it painted?
            if (reference.skeleton.at<uchar>(location.y(), location.x()) < 255) {
                Vector3d test;
                double dist = Camera::intersect(lead.cam->origin, lead.cam->ray(leadPixel).normalized(),
                                                reference.cam->origin, reference.cam->ray(location).normalized(),
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

// a combination that we already know correlates, but we keep it for later
struct candidate {
    imageData& lead;
    imageData& reference;
    Vector2d leadPixel, refPixel;
    Vector3d position;
    arteryNode& node;
};



//traces a line, starting from a pixel - correlating two renders / skeletons
//adds to a graph, appending at the node passed as
//marking traced pixels with 100, spawns new traces if other directions are present
//Pre: both imageData have the skeleton populated with black, g
void trace(imageData* lead, imageData* reference, Vector2d leadPixel, Vector2d refPixel,
           arteryGraph& graph, arteryNode* node, std::vector<candidate>& candidates){
    // make sure we are not doing bullshit
    assert(     lead->skeleton.at<uchar>(leadPixel.y(), leadPixel.x()) < 255);
    assert(reference->skeleton.at<uchar>( refPixel.y(),  refPixel.x()) < 255);

    // add the current pixel, correlated
    Vector3d position;
    Vector2d bestPixel;

    int added = 0;

    // add new candidates from lead pixel
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            Vector2d location = leadPixel + Vector2d(i,j);
            // unchecked pixel?
            if (lead->skeleton.at<uchar>(location.y(), location.x()) == 0) {
                double distance;
                int radius = correlate(*lead, *reference, leadPixel, refPixel, bestPixel, position, distance);
                if(radius < 2 && distance < 0.02) {
                    candidates.push_back(candidate{*lead, *reference, location, bestPixel, position, *node});
                    lead->skeleton.at<uchar>(location.y(), location.x()) = 80;
                    if (reference->skeleton.at<uchar>(bestPixel.y(), bestPixel.x()) < 100)
                        reference->skeleton.at<uchar>(bestPixel.y(), bestPixel.x()) = 81;
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
            if (reference->skeleton.at<uchar>(location.y(), location.x()) == 0) {
                double distance;
                int radius = correlate(*reference, *lead, refPixel, leadPixel, bestPixel, position, distance);
                if(radius < 2 && distance < 0.02) {
                    candidates.push_back(candidate{*reference, *lead, location, bestPixel, position, *node});
                    reference->skeleton.at<uchar>(location.y(), location.x()) = 82;
                    if (lead->skeleton.at<uchar>(bestPixel.y(), bestPixel.x()) < 100)
                        lead->skeleton.at<uchar>(bestPixel.y(), bestPixel.x()) = 83;
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
    candidate candy = candidates.back(); candidates.pop_back();
    arteryNode* node = candy.node.addNode(candy.position);
    candy.lead.skeleton.at<uchar>(candy.leadPixel.y(), candy.leadPixel.x()) = 100;
    candy.reference.skeleton.at<uchar>( candy.refPixel.y(),  candy.refPixel.x()) = 100;

    trace(&candy.lead, &candy.reference, candy.leadPixel, candy.refPixel, graph, node, candidates);
}

imageData data1, data2;

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
    data1.ray = data1.cam->ray(Vector2d(x,y)).normalized();
    Eigen::Vector4d line = data2.cam->projectLine(data1.cam->origin, data1.ray);
    data2.renderLine(line);

    data1.pointRdy = true;
    data1.executeRdy = true;

    Vector3d point;
    if (data2.pointRdy) {
        double distance = Camera::intersect(data1.cam->origin, data1.ray, data2.cam->origin, data2.ray, point);

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

    imshow("Cam1 Source", data1.visualisation);
    imshow("Cam2 Source", data2.visualisation);
}

static void onMouse2(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    data2.pixel = Vector2d(x,y);
    data2.ray = data2.cam->ray(Vector2d(x,y)).normalized();
    Eigen::Vector4d line = data1.cam->projectLine(data2.cam->origin, data2.ray);
    data1.renderLine(line);

    data2.pointRdy = true;
    data2.executeRdy = true;

    Vector3d point;
    if (data1.pointRdy) {
        double distance = Camera::intersect(data1.cam->origin, data1.ray, data2.cam->origin, data2.ray, point);
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

    imshow("Cam1 Source", data1.visualisation);
    imshow("Cam2 Source", data2.visualisation);
}

static void startTrace(int i, void* a) {
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
                if (data1.skeleton.at<uchar>(location.y(), location.x()) == 0) {
                    startLead = location;
                    foundLead = true;
                    goto endLead;
                }
            }
            }
            else {
                Vector2d location = data1.pixel + Vector2d(i, -k);
                if (data1.skeleton.at<uchar>(location.y(), location.x()) == 0) {
                    startLead = location;
                    foundLead = true;
                    goto endLead;
                }
                location = data1.pixel + Vector2d(i, +k);
                if (data1.skeleton.at<uchar>(location.y(), location.x()) == 0) {
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
                    if (data2.skeleton.at<uchar>(location.y(), location.x()) == 0) {
                        startRef = location;
                        foundRef = true;
                        goto endRef;
                    }
                }
            }
            else {
                Vector2d location = data2.pixel + Vector2d(i, -k);
                if (data2.skeleton.at<uchar>(location.y(), location.x()) == 0) {
                    startRef = location;
                    foundRef = true;
                    goto endRef;
                }
                location = data2.pixel + Vector2d(i, +k);
                if (data2.skeleton.at<uchar>(location.y(), location.x()) == 0) {
                    startRef = location;
                    foundRef = true;
                    goto endRef;
                }
            }

        }
    }
    endRef:

    imshow("Skeleton 1", data1.skeleton);
    imshow("Skeleton 2", data2.skeleton);

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
    imshow("Cam1 Source", data1.visualisation);
    imshow("Cam2 Source", data2.visualisation);

    int count = 0;

    while (!candidates.empty()) {
        count++;
        data1.renderPoint(candidates.back().position);
        data2.renderPoint(candidates.back().position);
        exploreOne(candidates, graph);
        imshow("Cam1 Source", data1.visualisation);
        imshow("Cam2 Source", data2.visualisation);

        imshow("Skeleton 1", data1.skeleton);
        imshow("Skeleton 2", data2.skeleton);
        //cv::waitKey(1);
    }
    return;
}

int main( int argc, char** argv )
{
    // folder
    std::string folder = "../data/multi_cam/scene02/";

    Camera cam1 = Camera(folder + "meta", 0);
    Camera cam2 = Camera(folder + "meta", 1);

    data1.cam = &cam1;
    data2.cam = &cam2;

    // ingest source images
    data1.source = imread( folder + cam1.name + ".png", cv::IMREAD_GRAYSCALE );
    data2.source = imread( folder + cam2.name + ".png", cv::IMREAD_GRAYSCALE );

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

    imshow( "Cam1 Source", data1.source);
    imshow( "Cam2 Source", data2.source );
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

void Skeletonize( imageData& data ) {

    data.source.copyTo(data.skeleton);

    // blur
    if (smooth) {
        cv::GaussianBlur(data.skeleton, data.skeleton, cv::Size(11,11), 0);
    }


    if (b_threshold) {
        cv::threshold(data.skeleton, data.skeleton, threshold, max_threshold, cv::ThresholdTypes::THRESH_BINARY);
    }

    if (dilate) {
        cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));

        cv::dilate(data.skeleton, data.skeleton, element);
    }

    if (b_thin) {
        //thin(target, smooth, acute_angle, destair); //about 2fps at 1k resolution
        cv::bitwise_not(data.skeleton, data.skeleton);
        cv::ximgproc::thinning(data.skeleton, data.skeleton, cv::ximgproc::THINNING_GUOHALL); //about 6 fps at 1k resolution
        cv::bitwise_not(data.skeleton, data.skeleton);
    }
}

void Skeletonize( int, void* )
{
    auto t1 = std::chrono::high_resolution_clock::now();
    Skeletonize(data1);
    Skeletonize(data2);

    imshow( "Skeleton 1", data1.skeleton );
    imshow( "Skeleton 2", data2.skeleton );

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