#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <imageData2.h>

int kernel_radius = 11;

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
        f = from::distance;
        text.append("distance");
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

void applyInitKernel( int state, void*) {
    uint16_t inner_size_px = kernel_radius * 2 + 1;
    uint16_t outer_size_px = kernel_radius * 4 + 1;
    uint16_t delta_radius_px = kernel_radius;

    // outer border is negative
    auto outer = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(outer_size_px,outer_size_px));
    auto inner = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size2i(inner_size_px,inner_size_px));
    cv::copyMakeBorder(inner, inner, delta_radius_px, delta_radius_px, delta_radius_px, delta_radius_px, cv::BORDER_CONSTANT, 0);
    outer.convertTo(outer, CV_8U);
    inner.convertTo(inner, CV_8U);

    // max 2, min 0
    cv::Mat kernel = 2 * inner + 0.8 - outer;
    
    // max 1, min 0
    kernel.convertTo(kernel, CV_32F, 0.5, 0);
    imshow( "Kernel", kernel);
    kernel -= 0.4;
    kernel *= 2;
    // make sure the best response is a 1
    float factor = 1. / (M_PI * kernel_radius * kernel_radius);
    kernel *= factor;

    cv::filter2D(data1.threshold[data1.visual_frame], data1.initConv[data1.visual_frame], 0, kernel);
    cv::filter2D(data2.threshold[data1.visual_frame], data2.initConv[data1.visual_frame], 0, kernel);

    displayVisual();
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
    cv::createButton("Apply Kernel", applyInitKernel);
    cv::createTrackbar( "Frame:", "", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Vis. Src:", "", &what, 6, changeVisual);
    cv::createTrackbar( "Kernel Size", "", &kernel_radius, 21, nullptr);

    // find best starting point:
    // create filters with a circle in the middle, the rest is negative
    applyInitKernel(0, nullptr);

    // just for one image now, largest response
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(data1.initConv[data1.visual_frame], &min, &max, &min_loc, &max_loc);
    
    // init values: 
    cv::Point location = max_loc;
    double angle = 0;
    double size = 10;


    cv::waitKey(0);
    return 0;
}