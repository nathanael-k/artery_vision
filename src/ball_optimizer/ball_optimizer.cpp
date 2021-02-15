#include "opencv2/imgproc.hpp"

#include <ball_optimizer.h>

BallOptimizer::BallOptimizer(Ball& ball, const StereoCamera& stereo_camera) 
    : ball(ball), stereo_camera(stereo_camera) {};

void BallOptimizer::optimize(const uint16_t steps, const uint8_t frame_index) {
    // project initial coordinates of ball

};

void BallOptimizer::step(const double dx, const uint8_t frame_index) {
    // calculate CircleGradients for both cameras
    CircleGradient grad0 = get_gradient(0, frame_index);
    CircleGradient grad1 = get_gradient(1, frame_index);

    // apply location adjustment 

    // apply rotation of ball direction

    // apply scale adjustment

}

Circle BallOptimizer::project_ball(uint8_t camera_index) const {
return Circle({0,0}, 10, 0);
}

CircleGradient BallOptimizer::get_gradient(uint8_t camera_index, const uint8_t frame_index) const {
    // where is the circle now?
    Circle circle = project_ball(camera_index);    
    
    // setting:
    uint8_t derivative_kernel_size = 31;

    cv::Point location = {(int)circle.location_px.x(), (int) circle.location_px.y()};
    double angle_deg = circle.angle_deg();
    double angle_rad = angle_deg / 180. * M_PI;
    // estimate radius from distance transform...
    double& radius = circle.radius_px;
    
    // source rectangle
    cv::Mat patch;

    double sinacosa = abs(sin(angle_rad)) + abs(cos(angle_rad));
    uint32_t edge_2 = ceil((6 * radius * sinacosa)/2);

    // if necessary, take larger source patch
    if (edge_2 < 16)
        edge_2 = 16;

    cv::Rect src_area = cv::Rect(location - cv::Point(edge_2,edge_2), location + cv::Point(edge_2,edge_2));
    auto src_patch = stereo_camera.image_data(camera_index).threshold[frame_index](src_area);

    // rotate
    double scale = derivative_kernel_size/(6*radius);
    cv::Mat derot;
    cv::Point center = cv::Point(edge_2, edge_2);
    auto rot = cv::getRotationMatrix2D(center, angle_deg, scale);
    cv::warpAffine(src_patch, derot, rot, cv::Size(2 * edge_2, 2 * edge_2));

    // only use center
    float border = (derot.cols - derivative_kernel_size)/2;
    auto rect = cv::Rect(border, border, derivative_kernel_size, derivative_kernel_size);
    patch = derot(rect);

    // inverse
    cv::Mat patch_inv = 1 - patch;

    // gaussian weighting
    cv::Mat gauss_kernel = 100 * (cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F) * cv::getGaussianKernel(derivative_kernel_size, -1, CV_32F).t());

    patch = gauss_kernel.mul(patch);
    patch_inv = gauss_kernel.mul(patch_inv);

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
    
    // calculate rotation gradient
    double inner_rot_grad = inner_rot.cwiseProduct(area_inv).sum();
    double outer_rot_grad = outer_rot.cwiseProduct(area).sum();

    // calculate scale gradient
    double scale_1_grad = scale_1.cwiseProduct(area_inv).sum();
    double scale_2_grad = scale_2.cwiseProduct(area).sum();

    // calculate translation gradient    
    double translation_1_grad = translation_1.cwiseProduct(area_inv).sum();
    double translation_2_grad = translation_2.cwiseProduct(area).sum();
    
    // return the gradient
    return {
        quality,
        inner_rot_grad, outer_rot_grad,
        scale_1_grad, scale_2_grad,
        translation_1_grad, translation_2_grad,
        };
}

void BallOptimizer::triangulate_circles(const Circle& circle_A, const Circle& circle_B) {
    const Camera& cam_A = stereo_camera.camera_A;
    const Camera& cam_B = stereo_camera.camera_B;
    
    // position
    Vector3d center;
    double distance = stereo_camera.triangulate(circle_A.location_px, circle_B.location_px, center);

    // radius
    double radius_A = cam_A.estimate_radius_world_m(center, circle_A.radius_px);
    double radius_B = cam_B.estimate_radius_world_m(center, circle_B.radius_px);

    // assert the radii do not disagree too much
    assert(abs(radius_A / radius_B - 1) < 0.3);

    double radius = ( radius_A + radius_B ) * 0.5;

    // direction

    // create a normed vector for each camera that would be the direction if perfectly perpendicular
    // then add and norm them for the initial direction estimate

    // we can just rotate the up vector around the direction
    Eigen::Vector3d dir_A = Eigen::AngleAxisd(circle_A.angle_rad(), cam_A.direction) * cam_A.cameraUp;
    Eigen::Vector3d dir_B = Eigen::AngleAxisd(circle_B.angle_rad(), cam_B.direction) * cam_B.cameraUp;

    // it can happen that they point in different directions
    double dot = dir_A.dot(dir_B);

    if (dot < 0)
        dir_B *= -1;

    Eigen::Vector3d direction = (dir_A + dir_B).normalized();

    ball.center_m = center;
    ball.radius_m = radius;
    ball.direction = direction;
    ball.confidence = 1 / (distance + 0.1);
}
