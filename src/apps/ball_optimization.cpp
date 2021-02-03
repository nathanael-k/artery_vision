#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <imageData2.h>

imageData data1("../data/renders/aorta_to_brain/", 0),
          data2("../data/renders/aorta_to_brain/", 1);

void displayVisual( int, void* )
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
        f = from::new_skeleton;
        text.append("new_skeleton");
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
    if (state == -1) {
        data1.apply_threshold(data1.visual_frame, 128, 255);
        data2.apply_threshold(data1.visual_frame, 128, 255);
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
    cv::namedWindow( "Control");
    cv::namedWindow( "Cam1 Visual", cv::WINDOW_AUTOSIZE);
    cv::namedWindow( "Cam2 Visual", cv::WINDOW_AUTOSIZE);
    imshow( "Cam1 Visual", (*data1.curr_displayed)[data1.visual_frame]);
    imshow( "Cam2 Visual", (*data2.curr_displayed)[data1.visual_frame]);

    int what = 0;

    cv::createTrackbar( "Frame:", "Control", &data1.visual_frame, data1.size-1, displayVisual);
    cv::createTrackbar( "Visualisation Source:", "Control", &what, 6, changeVisual);

    cv::createButton("Control", thresholdCurrent);

    // prepare skeletons
    //data1.prepareAllLayers();
    //data2.prepareAllLayers();

    // buildGraph(data1, data2);

    // input starting location to process (will be the position of catheter tip later on)
    // register mouse callback
    //cv::setMouseCallback("Cam1 Source", onMouse1);
    //cv::setMouseCallback("Cam2 Source", onMouse2);

    cv::waitKey(0);
    return 0;
}