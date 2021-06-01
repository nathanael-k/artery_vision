//
// Created by nate on 19.09.20.
//

#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "stereo_camera.h"

#include <Eigen/src/Core/Matrix.h>
#include <ball.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <imageData2.h>
#include <sys/types.h>

cv::Mat circleKernel(uint16_t kernel_radius)
{
  uint16_t inner_size_px = kernel_radius * 2 + 1;
  uint16_t outer_size_px = kernel_radius * 4 + 1;
  uint16_t delta_radius_px = kernel_radius;

  // outer border is negative
  auto outer = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size2i(outer_size_px, outer_size_px));
  auto inner = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size2i(inner_size_px, inner_size_px));
  cv::copyMakeBorder(inner, inner, delta_radius_px, delta_radius_px,
                     delta_radius_px, delta_radius_px, cv::BORDER_CONSTANT, 0);
  outer.convertTo(outer, CV_8U);
  inner.convertTo(inner, CV_8U);

  // max 2, min 0
  cv::Mat kernel = 2 * inner + 0.8 - outer;

  // max 1, min 0
  kernel.convertTo(kernel, CV_32F, 0.5, 0);
  imshow("Kernel", kernel);
  kernel -= 0.4;
  kernel *= 2;
  // make sure the best response is a 1
  float factor = 1. / (M_PI * kernel_radius * kernel_radius);
  kernel *= factor;

  return kernel;
}

void maxPoolCircleConvolution(const cv::Mat &source, cv::Mat &destination)
{
  // max of convolutions of different filter sizes
  cv::Mat buffer = destination;
  cv::filter2D(source, destination, -1, circleKernel(7));

  for (int i = 8; i < 30; i++)
  {
    cv::filter2D(source, buffer, -1, circleKernel(i));
    destination = cv::max(buffer, destination);
  }
}

void initCircleCenters(const cv::Mat &source, cv::Mat &destination)
{
  // find promising spots
  maxPoolCircleConvolution(source, destination);
  // find local maxima
  cv::Mat buffer;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::dilate(destination, buffer, element);
  cv::Mat mask;
  cv::compare(buffer, destination, mask, cv::CMP_EQ);
  cv::bitwise_and(destination, mask, destination);
}

imageData::imageData(std::string metaFolder, const Camera &camera)
    : cam(camera)
{
  // read size from file
  std::ifstream inFile;
  inFile.open(metaFolder + "meta");
  if (!inFile)
  {
    std::cerr << "Unable to open file: " << metaFolder;
    exit(1); // call system to stop
  }
  // skip 1 line
  std::string textBuffer;
  std::getline(inFile, textBuffer);
  // getSize
  inFile >> size;
  // read all files
  for (int i = 0; i < size; i++)
  {
    char buffer[50];
    int n = sprintf(buffer, "%03d", i);
    assert(n < 50);
    std::string file = metaFolder + cam.name + '_' + buffer + ".png";
    source.push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
  }
  visualisation.resize(size);
  endpoints.resize(size);
  initConv.resize(size);
  threshold.resize(size);
  buffer.resize(size);
  distance.resize(size);
  for (int i = 0; i < size; i++)
  {
    // THRESHOLDING might have to be adjusted depending on source images
    cv::threshold(source[i], threshold[i], 128, 255, cv::THRESH_BINARY_INV); // rendered
    //cv::threshold(source[i], threshold[i], 140, 255, cv::THRESH_BINARY_INV); // printed
    source[i].copyTo(visualisation[i]);
    source[i].copyTo(endpoints[i]);
    source[i].copyTo(buffer[i]);
    cv::distanceTransform(threshold[i], distance[i], cv::DIST_L2,
                          cv::DIST_MASK_3);
    initCircleCenters(threshold[i], initConv[i]);
    //
  }
}

void imageData::resetVisual(from where, int frame)
{
  if (where == from::source)
  {
    curr_displayed = &source;
  }
  if (where == from::threshold)
  {
    if (!threshold[frame].empty())
    {
      curr_displayed = &threshold;
    }
  }
  if (where == from::initConv)
  {
    if (!initConv[frame].empty())
    {
      curr_displayed = &initConv;
    }
  }
  if (where == from::visualisation)
  {
    if (!visualisation[frame].empty())
    {
      curr_displayed = &visualisation;
    }
  }
  if (where == from::endpoints)
  {
    if (!endpoints[frame].empty())
    {
      curr_displayed = &endpoints;
    }
  }
  if (where == from::buffer)
  {
    if (!buffer[frame].empty())
    {
      curr_displayed = &buffer;
    }
  }
  if (where == from::distance)
  {
    if (!distance[frame].empty())
    {
      curr_displayed = &distance;
    }
  }
}

void imageData::renderLine(const Eigen::Vector4d &line, int index)
{
  assert(index < visualisation.size());
  cv::Point A(line[0], line[1]);
  cv::Point B(line[2], line[3]);
  cv::line(visualisation[index], A, B, CV_RGB(100, 100, 100));
}

void imageData::renderLine(const Eigen::Vector3d &begin,
                           const Eigen::Vector3d &end, int index)
{
  assert(index < visualisation.size());
  Vector2d a, b;
  a = cam.projectPoint(begin);
  b = cam.projectPoint(end);
  cv::Point A(a[0], a[1]);
  cv::Point B(b[0], b[1]);
  cv::line(visualisation[index], A, B, CV_RGB(255, 100, 100));
}

void imageData::renderPoint(Vector2d point, int label, int index)
{
  cv::Point P(point[0], point[1]);
  cv::circle(visualisation[index], P, 3, CV_RGB(100, 100, 255), 2);
  if (label >= 0)
    cv::putText(visualisation[index], std::to_string(label), P, 0, 1,
                CV_RGB(255, 255, 255));
}

void imageData::renderPoint(Vector3d point, int label, int index)
{
  renderPoint(cam.projectPoint(point), label, index);
}

void imageData::renderBall(const Ball &ball, int index)
{
  Circle circle = project_circle(ball, cam);
  cv::circle(visualisation[index],
             cv::Point(circle.location_px.x(), circle.location_px.y()),
             circle.radius_px, CV_RGB(100, 100, 255), 1);
}

void imageData::renderNode(const arteryNode &node, int index)
{
  renderBall(node.ball, index);
  Eigen::Vector2d text_center = cam.projectPoint(node.ball.center_m);
  cv::putText(visualisation[index], std::to_string(node.index),
              cv::Point(text_center.x(), text_center.y()), 0, 1,
              CV_RGB(255, 255, 255));
}

int correlate(const cv::Mat &ref, const Camera &leadCam, const Camera &refCam,
              const Vector2d &leadPixel, const Vector2d &refPixel,
              Vector2d &bestPixel, Vector3d &point, double &distance, int range,
              int cutoff)
{

  distance = std::numeric_limits<double>::max();
  int pixelDistance = -1;

  // go trough whole neighbourhood
  for (int i = -range; i <= range; i++)
  {
    for (int j = -range; j <= range; j++)
    {
      Vector2d location = refPixel + Vector2d(i, j);
      // is it part of the skeleton?
      if (ref.at<uchar>(location.y(), location.x()) < cutoff)
      {
        Vector3d test;
        double dist =
            Camera::intersect(leadCam, leadPixel, refCam, refPixel, test);
        // is it closer?
        if (dist < distance)
        {
          bestPixel = location;
          distance = dist;
          point = test;
          // is the new point connected?
          pixelDistance = std::max(abs(i), abs(j));
        }
      }
    }
  }
  return pixelDistance;
}

Vector2d double_locate(cv::Mat &prio, cv::Mat &backup,
                       const cv::Mat &components, const Eigen::Vector4d &line,
                       const Vector2d pos, const Camera &cam_ref,
                       const Vector2d &pixel_ref, const Camera &cam_new)
{

  Vector2d res =
      locate(prio, components, line, pos, cam_ref, pixel_ref, cam_new);
  if (res == Vector2d(-1, -1))
  {
    res = locate(backup, components, line, pos, cam_ref, pixel_ref, cam_new);
  }
  assert(res != Vector2d(-1, -1));
  return res;
}

Vector2d locate(cv::Mat &ref, const cv::Mat &components,
                const Eigen::Vector4d &line, const Vector2d pos,
                const Camera &cam_ref, const Vector2d &pixel_ref,
                const Camera &cam_new)
{

  struct candidate
  {
    Vector2d location;
    int component;
    double distance;
  };
  std::vector<candidate> candidates;
  int weak_index;
  Vector3d dummy;

  // test multiple lines
  // for

  // we basically draw a line, but make sure that it is 4 connect or we could
  // miss a component!
  Vector2d start(line[0], line[1]);
  Vector2d end(line[2], line[3]);
  Vector2d delta = end - start;
  if (abs(delta.x()) > abs(delta.y()))
  {
    delta /= abs(delta.x());
    weak_index = 0;
  }
  else
  {
    delta /= abs(delta.y());
    weak_index = 1;
  }

  Vector2d side = delta.normalized();

  for (int off = -3; off <= 3; off++)
  {

    Vector2d curr = start;
    curr.y() += off * side.x() * 15;
    int curr_weak = curr[weak_index];

    for (int i = 0; i < ref.cols; i++)
    {
      if (ref.at<uchar>(curr.y(), curr.x()) > 200)
        candidates.push_back(
            {curr, components.at<uchar>(curr.y(), curr.x()),
             Camera::intersect(cam_ref, pixel_ref, cam_new,
                               Vector2d(curr.x(), curr.y()), dummy)});
      ref.at<uchar>(curr.y(), curr.x()) = 99;
      // debug draw to see something
      curr += delta;
      // 4 connected!
      if (curr_weak != curr[weak_index])
      {
        Vector2d buff = curr;
        buff[weak_index] = curr_weak;
        if (ref.at<uchar>(buff.y(), buff.x()) > 200)
          candidates.push_back(
              {curr, components.at<uchar>(buff.y(), buff.x()),
               Camera::intersect(cam_ref, pixel_ref, cam_new,
                                 Vector2d(buff.x(), buff.y()), dummy)});
        curr_weak = curr[weak_index];
        ref.at<uchar>(buff.y(), buff.x()) = 100;
      }
    }
  }

  Vector2d best;

  if (candidates.size() == 0)
    return Vector2d(-1, -1);

  double distance = std::numeric_limits<double>::max();
  int best_component;
  for (auto &candy : candidates)
  {
    double dist = (candy.location - pos).norm();
    if (dist < distance)
    {
      best_component = candy.component;
      distance = dist;
    }
  }

  // on the closest component, find the best match
  distance = std::numeric_limits<double>::max();
  for (auto &candy : candidates)
  {
    if (candy.component == best_component)
    {

      if (candy.distance < distance)
      {
        best = candy.location;
        distance = candy.distance;
      }
    }
  }

  return best;
}

Circle initialize_Circle(const cv::Point &coord, const cv::Mat &distances,
                         const cv::Mat &threshold)
{
  // check if coordinate is even valid according to distance map
  assert(threshold.at<uint8_t>(coord) > 0);

  // select radius based on distance information
  float radius = distances.at<float>(coord);
  assert(radius > 3);

  std::vector<Circle> circles =
      find_adjacent_circles(coord, radius * 1.5, distances, threshold);

  Circle ret(Eigen::Vector2d(coord.x, coord.y), radius, 0);

  ret.connections = circles.size();
  if (circles.size() > 0)
  {
    ret.point_at(circles[0].location_px);
  }

  return ret;
}

std::vector<Circle> find_adjacent_circles(const cv::Point &coord,
                                          uint8_t radius,
                                          const cv::Mat &distances,
                                          const cv::Mat &threshold)
{
  std::vector<Circle> ret;
  if (threshold.at<uint8_t>(coord) == 0)
    return ret;

  // generate an array with values that corresponds to the circumference
  std::vector<cv::Point2i> coordinates;
  fill_circle_coordinates(coordinates, coord, radius);
  std::vector<u_int8_t> circle_values;
  fill_array(circle_values, coordinates, threshold);

  size_t start = 0;
  size_t count = circle_values.size();

  // count connections

  // we want to start at a 0, if we have 1s at the beginning we put them at the
  // back
  while (circle_values[start] > 0 && start < count)
  {
    circle_values.emplace_back(circle_values[start]);
    start++;
  }

  // append a 0 so that the following loop ends by closing connection
  circle_values.emplace_back(circle_values[start]);

  bool in_connection = false;
  size_t connection_start = start;
  float max_distance = 0;
  size_t max_index;

  for (size_t index = start; index < circle_values.size(); index++)
  {
    if (circle_values[index] == 0)
    {
      if (in_connection)
      {
        // close connection and generate circle
        // center of the circle is the maximum value of distances

        Eigen::Vector2d position_px(coordinates[max_index].x,
                                    coordinates[max_index].y);
        float radius_px = max_distance;

        // only add if we actually have some radius
        if (radius_px > 0.2 * radius)
        {
          ret.emplace_back(position_px, radius_px,
                           Eigen::Vector2d(coord.x, coord.y));
        }
        in_connection = false;
        max_distance = 0;
      }
    }
    else
    {
      // we have a one
      size_t capped_index = index % count;
      float distance = distances.at<float>(coordinates[capped_index]);
      if (distance > max_distance)
      {
        max_distance = distance;
        max_index = capped_index;
      }
      if (!in_connection)
      {
        in_connection = true;
        connection_start = index;
      }
    }
  }

  return ret;
}

std::vector<Circle> find_adjacent_circles(const Circle &circle,
                                          const double radius_factor,
                                          const cv::Mat &distances,
                                          const cv::Mat &threshold)
{
  cv::Point2i center(circle.location_px.x(), circle.location_px.y());
  return find_adjacent_circles(center, circle.radius_px * radius_factor,
                               distances, threshold);
}

void fill_circle_coordinates(std::vector<cv::Point2i> &coordinates,
                             cv::Point2i center, u_int8_t radius)
{
  // Bresenham
  std::vector<cv::Point2i> first_octant;
  int x = 0;
  int y = radius;
  int d = 3 - 2 * radius;
  first_octant.emplace_back(cv::Point(x, y));
  while (y > x)
  {
    x++;

    if (d > 0)
    {
      y--;
      d = d + 4 * (x - y) + 10;
    }
    else
      d = d + 4 * x + 6;

    first_octant.emplace_back(cv::Point(x, y));
  }
  // first octant is filled;
  coordinates.clear();
  coordinates.reserve(first_octant.size() * 8);

  // copy into octants and apply center translation
  // top right quadrant is +x -y
  for (int i = 0; i < first_octant.size(); i++)
  {
    coordinates.emplace_back(center.x + first_octant[i].x,
                             center.y - first_octant[i].y);
  }
  for (int i = first_octant.size() - 2; i >= 0; i--)
  {
    coordinates.emplace_back(center.x + first_octant[i].y,
                             center.y - first_octant[i].x);
  }

  // bottom right quadrant is +x +y
  for (int i = 1; i < first_octant.size(); i++)
  {
    coordinates.emplace_back(center.x + first_octant[i].y,
                             center.y + first_octant[i].x);
  }
  for (int i = first_octant.size() - 2; i >= 0; i--)
  {
    coordinates.emplace_back(center.x + first_octant[i].x,
                             center.y + first_octant[i].y);
  }

  // bottom left quadrant is -x +y
  for (int i = 1; i < first_octant.size(); i++)
  {
    coordinates.emplace_back(center.x - first_octant[i].x,
                             center.y + first_octant[i].y);
  }
  for (int i = first_octant.size() - 2; i >= 0; i--)
  {
    coordinates.emplace_back(center.x - first_octant[i].y,
                             center.y + first_octant[i].x);
  }

  // top left quadrant is -x -y
  for (int i = 1; i < first_octant.size(); i++)
  {
    coordinates.emplace_back(center.x - first_octant[i].y,
                             center.y - first_octant[i].x);
  }
  for (int i = first_octant.size() - 2; i >= 1; i--)
  {
    coordinates.emplace_back(center.x - first_octant[i].x,
                             center.y - first_octant[i].y);
  }

  // all coordinates collected
}

bool in_bounds(cv::Point point, const cv::Mat &mat)
{
  if (point.x < 0 || point.y < 0)
    return false;
  if (point.x >= mat.cols || point.y >= mat.rows)
    return false;

  return true;
}

void fill_array(std::vector<u_int8_t> &array,
                const std::vector<cv::Point2i> &coordinates,
                const cv::Mat &source)
{
  array.clear();
  array.reserve(coordinates.size());
  for (const auto &coord : coordinates)
  {
    if (in_bounds(coord, source))
      array.emplace_back(source.at<uint8_t>(coord));
    else
      array.emplace_back(0);
  }
}

std::vector<Circle> extract_init_circles(size_t count, const cv::Mat &init,
                                         const cv::Mat &threshold,
                                         const cv::Mat &distances)
{
  cv::Size dim = init.size();
  cv::Mat suppression_mask = cv::Mat(dim, CV_8UC1, 255);

  std::vector<Circle> ret;

  for (int i = 0; i < count; i++)
  {

    // find highest point
    double min, max;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(init, &min, &max, &minLoc, &maxLoc, suppression_mask);

    // create circle
    if (threshold.at<uint8_t>(maxLoc) > 0 && distances.at<float>(maxLoc) > 3)
      ret.emplace_back(initialize_Circle(maxLoc, distances, threshold));

    float radius = distances.at<float>(maxLoc);

    // supress vicinity
    cv::circle(suppression_mask, maxLoc, radius * 2, 0, cv::FILLED);
  }
  // return
  return ret;
}
