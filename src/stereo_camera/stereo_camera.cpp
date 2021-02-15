#include <stereo_camera.h>

StereoCamera::StereoCamera(std::string meta_folder) 
    : camera_A(meta_folder+"meta", 0), camera_B(meta_folder+"meta", 1),
    image_data_A(meta_folder, camera_A), image_data_B(meta_folder, camera_B) {
    total_frames = image_data_A.size;
    assert(image_data_A.size == image_data_B.size);
}

double StereoCamera::triangulate(const Eigen::Vector2d& point_cam_A, const Eigen::Vector2d& point_cam_B, Eigen::Vector3d& out_intersection) {
    // rely on camera implementation
    return Camera::intersect(camera_A, point_cam_A, camera_B, point_cam_B, out_intersection);
}

const imageData& StereoCamera::image_data(uint8_t index) const {
    if (index == 0)
        return image_data_A;
    if (index == 1)
        return image_data_B;
    
    // invalid index
    assert (false);
}

const cv::Mat& StereoCamera::displayed_image_A() const {
    return (*image_data_A.curr_displayed)[current_displayed_frame];
}

const cv::Mat& StereoCamera::displayed_image_B() const {
    return (*image_data_B.curr_displayed)[current_displayed_frame];
}

void StereoCamera::reset_visual(from where) {
    image_data_A.resetVisual(where, current_displayed_frame);
    image_data_B.resetVisual(where, current_displayed_frame);
}