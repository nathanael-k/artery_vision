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

