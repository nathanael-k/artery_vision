#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <imageData2.h>

imageData data1("../data/renders/aorta_to_brain/", 0),
          data2("../data/renders/aorta_to_brain/", 1);

void displayVisual( int = 0, void* =  nullptr )
{
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);
}

void changeVisual( int pos, void* )
{

    std::string text = "Image source: ";
    from f;
    if (pos == 0) {
        f = from::source;
        text.append("source");
    }
    if (pos == 1) {
        f = from::threshold;
        text.append("threshold");
    }
    if (pos == 2) {
        f = from::initConv;
        text.append("initConv");
    }
    if (pos == 3) {
        f = from::components;
        text.append("components");
    }
    if (pos == 4) {
        f = from::endpoints;
        text.append("endpoints");
    }
    if (pos == 5) {

        f = from::buffer;
        text.append("buffer");
    }
    if (pos == 6) {
        f = from::visualisation;
        text.append("visualisation");
    }

    data1.resetVisual(f);
    data2.resetVisual(f);
    cv::displayStatusBar("Control", text);
    displayVisual(0,0);
}

void thresholdCurrent( int state, void*) {
    // if button got pressed down
    if (state == 0) {
        data1.apply_threshold(data1.visual_frame, 128, 255);
        data2.apply_threshold(data1.visual_frame, 128, 255);
        displayVisual();
    }
}

int main( int argc, char** argv )
{

    if( data1.source.empty() ||  data2.source.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        return -1;
    }

    // create windows
    // cv::namedWindow( "Control");
    cv::namedWindow( "Cam1 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Kernel", cv::WINDOW_AUTOSIZE);
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);

    int what = 0;


    cv::createButton("Threshold", thresholdCurrent);
    cv::createTrackbar( "Frame:", "", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Vis. Src:", "", &what, 6, changeVisual);

    // find best starting point:
    // create filters with a circle in the middle, the rest is negative
    
    uint16_t inner_radius_px = 21;
    uint16_t outer_radius_px = 31;
    uint16_t delta_radius_px = outer_radius_px-inner_radius_px;

    // outer border is negative
    auto outer = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(outer_radius_px,outer_radius_px));
    auto inner = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(inner_radius_px,inner_radius_px));
    cv::copyMakeBorder(inner, inner, delta_radius_px/2, delta_radius_px/2, delta_radius_px/2, delta_radius_px/2, cv::BORDER_CONSTANT, 0);
    outer.convertTo(outer, CV_8U);
    inner.convertTo(inner, CV_8U);

    cv::Mat kernel = 2 * inner + 1 - outer;
    float factor = 1. / static_cast<float>(outer_radius_px * outer_radius_px);
    kernel.convertTo(kernel, CV_32F, factor,-1);
    imshow( "Kernel", kernel);

    cv::filter2D((*data1.curr_displayed)[data1.visual_frame], data1.initConv[data1.visual_frame], 0, kernel);
    cv::filter2D((*data1.curr_displayed)[data1.visual_frame], data1.initConv[data1.visual_frame], 0, kernel);


    cv::waitKey(0);
    return 0;
}