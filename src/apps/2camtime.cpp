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
#include <imageData.h>

imageData data1("../data/renders/flow_1/", 0),
          data2("../data/renders/flow_1/", 1);

void displayVisual( int, void* );
void changeVisual( int pos, void* );

int main( int argc, char** argv )
{
    if( data1.source.empty() ||  data2.source.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }


    // create windows
    cv::namedWindow( "Control");
    cv::namedWindow( "Cam1 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Visual", cv::WINDOW_AUTOSIZE);
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);

    int what = 0;

    cv::createTrackbar( "Frame:", "Control", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Visualisation Source:", "Control", &what, 4, changeVisual);

    // prepare skeletons
    data1.SkeletonizeAll();
    data2.SkeletonizeAll();

    // prepare endpoints
    data1.EndpointsAll();
    data2.EndpointsAll();

    // input starting location to process (will be the position of catheter tip later on)
    // register mouse callback
    //cv::setMouseCallback("Cam1 Source", onMouse1);
    //cv::setMouseCallback("Cam2 Source", onMouse2);

    cv::waitKey(0);
    return 0;
}

void displayVisual( int, void* )
{
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
}

void changeVisual( int pos, void* )
{
    from f;
    if (pos == 0)
        f = from::source;
    if (pos == 1)
        f = from::skeleton;
    if (pos == 2)
        f = from::endpoints;
    if (pos == 3)
        f = from::buffer;
    if (pos == 4)
        f = from::visualisation;

    data1.resetVisual(f);
    data2.resetVisual(f);
    displayVisual(0,0);
}

