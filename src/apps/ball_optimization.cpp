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
    cv::Point location = max_loc + cv::Point(-3,0);
    uint8_t derivative_kernel_size = 31;
    double angle_deg = -32;
    double angle_rad = angle_deg / 180. * M_PI;
    // estimate radius from distance transform...
    float radius = data1.distance[data1.visual_frame].at<float>(location);
    
    // source rectangle
    cv::Mat patch;

    double sinacosa = abs(sin(angle_rad)) + abs(cos(angle_rad));
    uint32_t edge_2 = ceil((6 * radius * sinacosa)/2);



    // TODO: if necessary, add border (pad). or take larger source patch
    if (edge_2 < 16)
        edge_2 = 16;

    cv::Rect src_area = cv::Rect(location - cv::Point(edge_2,edge_2), location + cv::Point(edge_2,edge_2));
    auto src_patch = data1.threshold[data1.visual_frame](src_area);

    //imshow( "Kernel", src_patch);
    //cv::waitKey(0);

    // rotate
    double scale = derivative_kernel_size/(6*radius);
    cv::Mat derot;
    cv::Point center = cv::Point(edge_2, edge_2);
    auto rot = cv::getRotationMatrix2D(center, angle_deg, scale);
    cv::warpAffine(src_patch, derot, rot, cv::Size(2 * edge_2, 2 * edge_2));
    //imshow( "Kernel", derot);
    //cv::waitKey(0);

    // only use center
    float border = (derot.cols - derivative_kernel_size)/2;
    auto rect = cv::Rect(border, border, derivative_kernel_size, derivative_kernel_size);
    patch = derot(rect);

    imshow( "Kernel", patch);
    cv::waitKey(0);

    // inverse
    cv::Mat patch_inv = 1 - patch;
    //imshow( "Kernel", patch_inv);
    //cv::waitKey(0);

    // gaussian weighting
    cv::Mat gauss_kernel = 100 * (cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F) * cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F).t());
    //imshow( "Kernel", gauss_kernel );
    //cv::waitKey(0);

    patch = gauss_kernel.mul(patch);
    patch_inv = gauss_kernel.mul(patch_inv);


    //imshow( "Kernel", patch);
    //cv::waitKey(0);
    //imshow( "Kernel", patch_inv);
    //cv::waitKey(0);

    // calculate areas of interest
    cv::Rect A1, A2, A3, A4, B1, B2, B3, B4;
    A1 = cv::Rect(0,0,10,15); A2 = cv::Rect(10,0,5,15);A3 = cv::Rect(15,0,6,15);A4 = cv::Rect(21,0,10,15);
    B1 = cv::Rect(0,15,10,16); B2 = cv::Rect(10,15,5,16);B3 = cv::Rect(15,15,6,16);B4 = cv::Rect(21,15,10,16);
    
    Eigen::MatrixXd area(2,4);
    area << cv::sum(patch(A1))[0], cv::sum(patch(A2))[0], cv::sum(patch(A3))[0], cv::sum(patch(A4))[0],
            cv::sum(patch(B1))[0], cv::sum(patch(B2))[0], cv::sum(patch(B3))[0], cv::sum(patch(B4))[0];
 
    Eigen::MatrixXd area_inv(2,4);
    area_inv << cv::sum(patch_inv(A1))[0], cv::sum(patch_inv(A2))[0], cv::sum(patch_inv(A3))[0], cv::sum(patch_inv(A4))[0],
                cv::sum(patch_inv(B1))[0], cv::sum(patch_inv(B2))[0], cv::sum(patch_inv(B3))[0], cv::sum(patch_inv(B4))[0];
    
    // Selection matrices
    Eigen::MatrixXd background(2, 4);
    background << 1,0,0,1,
                  1,0,0,1;
    Eigen::MatrixXd artery(2,4);
    artery << 0,1,1,0,
              0,1,1,0;

    // Rotation selection matrices
    Eigen::MatrixXd inner_rot(2, 4);
    inner_rot << 0,1,-1,0,
                 0,-1,1,0;
    Eigen::MatrixXd outer_rot(2,4);
    outer_rot << -1,0,0,1,
                 1,0,0,-1;

    // Scale selection matrices
    Eigen::MatrixXd scale_1(2, 4);
    scale_1 << 0,0,0,0,
                 0,-1,-1,0;
    Eigen::MatrixXd scale_2(2,4);
    scale_2 << 0,0,0,0,
                 1,0,0,1;

    // Translation selection matrices
    Eigen::MatrixXd translation_1(2, 4);
    translation_1 << 0,1,-1,0,
                     0,1,-1,0;
    Eigen::MatrixXd translation_2(2,4);
    translation_2 << -1,0,0,1,
                     -1,0,0,1;

    // match quality is white pixels where the artery should be minus white pixels where the background should be
    double quality = (area.cwiseProduct(artery).sum() - area.cwiseProduct(background).sum()) / cv::sum(gauss_kernel)[0];
    
    // check rotation gradient
    double inner_rot_grad = inner_rot.cwiseProduct(area_inv).sum();
    double outer_rot_grad = outer_rot.cwiseProduct(area).sum();

    // check scale gradient
    double scale_1_grad = scale_1.cwiseProduct(area_inv).sum();
    double scale_2_grad = scale_2.cwiseProduct(area).sum();

    // check translation gradient    
    double translation_1_grad = translation_1.cwiseProduct(area_inv).sum();
    double translation_2_grad = translation_2.cwiseProduct(area).sum();
 
    cv::waitKey(0);
    return 0;
}