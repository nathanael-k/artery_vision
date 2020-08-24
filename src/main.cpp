#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include "zhangsuen.h"

#include <iostream>
#include <chrono>

cv::Mat src, source, target;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 4;
int threshold = 218;
int const max_elem = 2;
int const max_kernel_size = 21;
int const max_threshold = 255;

bool enable_update=false;
bool b_thin=true;
bool b_threshold=true;
bool dilate=true;
bool smooth=true;
bool acute_angle=true;
bool destair=true;

enum class button {
    dilate,
    smooth,
    acute_angle,
    destair,
    thin,
    threshold
};

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
    if (*where == button::dilate)            dilate = i;
    if (*where == button::smooth)           smooth = i;
    if (*where == button::acute_angle)      acute_angle = i;
    if (*where == button::destair)          destair = i;
    if (*where == button::thin)          b_thin = i;
    if (*where == button::threshold)          b_threshold = i;
    if (enable_update) Skeletonize(0,0);
};

int main( int argc, char** argv )
{
    cv::CommandLineParser parser( argc, argv, "{@input | LinuxLogo.jpg | input image}" );
    src = imread( parser.get<std::string>( "@input" ) , cv::IMREAD_GRAYSCALE );

    if( src.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    cv::namedWindow( "Source", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Skeleton Demo", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Endpoint Demo", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar( "Threshold:", "Skeleton Demo", &threshold, max_threshold, Skeletonize);
    cv::createTrackbar( "Kernel size:\n 2n +1", "Skeleton Demo",
                        &dilation_size, max_kernel_size,
                        Skeletonize );
    button choices[6] = {button::dilate, button::smooth, button::acute_angle, button::destair, button::thin, button::threshold};
    cv::createButton( "Dilate at beginning", callBackBtn, &choices[0], cv::QT_CHECKBOX, dilate);
    cv::createButton( "Boundary Smoothing", callBackBtn, &choices[1], cv::QT_CHECKBOX, smooth);
    cv::createButton( "Acute Angle Emphasis", callBackBtn, &choices[2], cv::QT_CHECKBOX, acute_angle);
    cv::createButton( "Destair", callBackBtn, &choices[3], cv::QT_CHECKBOX, destair);
    cv::createButton("Thin", callBackBtn, &choices[4], cv::QT_CHECKBOX, b_thin);
    cv::createButton("Threshold", callBackBtn, &choices[5], cv::QT_CHECKBOX, b_threshold);
    imshow( "Source", src );
    enable_update = true;
    Skeletonize(0, 0);

    cv::waitKey(0);
    return 0;
}

void Skeletonize( int, void* )
{
    auto t1 = std::chrono::high_resolution_clock::now();
    source = src;
    if (b_threshold) {
        cv::threshold(source, target, threshold, max_threshold, cv::ThresholdTypes::THRESH_BINARY);
        source = target;
    }

    if (dilate) {
        cv::Mat element = getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));

        cv::dilate(source, target, element);
        source = target;
    }

    if (b_thin) {
        thin(target, smooth, acute_angle, destair);
    }

    imshow( "Skeleton Demo", target );
    auto t2 = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::string text = cv::format("Time: %dms | FPS: %f", duration, (1000.0/duration));
    cv::displayStatusBar("Skeleton Demo", text);

    //Find endpoints: only one pixel in their 8 neighbourhood
    source = target.clone();
    cv::Mat endpoints = cv::Mat::zeros(src.cols, src.rows, src.type());
    std::cout << source.type();
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
    imshow( "Endpoint Demo", endpoints_visual );
    cv::displayStatusBar("Endpoint Demo", type2str(src.type()));

    // build graph
    // find first endpoint, this will be our first node

    // from that endpoint look in vicinity and continue on the path, creating a path_node every n pixels

    // until you hit something with more than two connections -> a junction -> a new node
    // iteratively proceed on both directions until you are at an endpoint

    // whenever you cant progress, check if it is actually an endpoint as decided earlier
}