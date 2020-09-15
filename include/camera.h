//
// Created by nate on 14.09.20.
//

#ifndef ARTERY_VISION_CAMERA_H
#define ARTERY_VISION_CAMERA_H


#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

using Eigen::Vector3d;
using Eigen::Vector2d;

class Camera {

public:
    std::string name;

    // in m
    Vector3d origin;

    Vector3d direction;
    Vector3d cameraUp;
    Vector3d cameraLeft;

    // position of pixel 0,0 on the image plane
    Vector3d topLeftPosition;

    // in m
    double sensorSize = 0;
    double focalLength = 0;

    // pixels
    double resolution = 0;

    // distance every pixel occupies on the sensor
    double pixelDistance;

    Camera() {

    }

    Camera(std::string metaPath, int index) {

        // read in from file
        std::ifstream inFile;
        inFile.open(metaPath);
        if (!inFile) {
            std::cerr << "Unable to open file: " << metaPath;
            exit(1);   // call system to stop
        }

        std::string textBuffer;
        int nCams;
        if (!(inFile >> nCams >> textBuffer)) {
            std::cerr << "Unable to read file: " << metaPath;
            exit(1);
        }

        if (textBuffer != "cameras") {
            std::cerr << "Unexpected string encountered while parsing " << metaPath << "\nString: " << textBuffer;
            exit(1);
        }

        if (index >= nCams) {
            std::cerr << "Cam index out of range: " << index << " : " << nCams;
            exit(1);
        }

        for (std::string line; std::getline(inFile, line);) {
            if (line == "cam") {
                int i;
                inFile >> i;
                if (i == index) {
                    inFile >> name;

                    inFile >> origin.x();
                    inFile >> origin.y();
                    inFile >> origin.z();

                    inFile >> direction.x();
                    inFile >> direction.y();
                    inFile >> direction.z();
                    direction.normalize();

                    inFile >> cameraUp.x();
                    inFile >> cameraUp.y();
                    inFile >> cameraUp.z();
                    cameraUp.normalize();

                    // direction and cameraUp must be orthogonal to each other
                    assert(cameraUp.cross(direction).norm() > 0.9999);

                    inFile >> focalLength;
                    inFile >> sensorSize;
                    inFile >> resolution;

                    // convert mm to m
                    focalLength /= 1000;
                    sensorSize /= 1000;

                    assert(focalLength > 0 && sensorSize > 0 && resolution > 0);

                    // orthogonal system
                    cameraLeft = -direction.cross(cameraUp);

                    // correct, if all directions are unit length
                    pixelDistance = (sensorSize / resolution);

                    topLeftPosition = focalLength*direction + (sensorSize*cameraUp + sensorSize*cameraLeft)/2;

                    return;
                }
            }
        }

        std::cerr << "Unable to find cam " << index << " in file " << metaPath;
        exit(1);
    }

    // origin is top left, x is right, y is down (openCV convention)
    // points from origin to the pixel on the image plane
    inline Vector3d ray(Vector2d position) {
        return (topLeftPosition - cameraLeft * position.x() * pixelDistance - cameraUp * position.y() * pixelDistance);
    }

    // returns the image coordinates (at least one of x or y >= 0) on success
    // returns -1 -1 on fail behind camera
    inline Vector2d projectPoint(Vector3d point) {
        // initialize as "fail" signal
        Vector2d ret = {-1,-1};

        // where does the ray from origin to point cross the image plane
        Vector3d pointRay = point-origin;

        // if point lies behind camera we dont project
        if (pointRay.dot(direction) <= 0)
            return ret;

        // construct image plane and line
        Eigen::ParametrizedLine<double,3> line = Eigen::ParametrizedLine<double,3> (origin, pointRay);
        Eigen::Hyperplane<double,3> imagePlane = Eigen::Hyperplane<double,3>(direction, origin + direction*focalLength);

        // intersect ray with image plane
        Vector3d projection = line.intersectionPoint(imagePlane);

        // delta is the relative vector on the imageplane, from top left pointing to our projected point
        Vector3d delta = projection-(topLeftPosition+origin);

        // x coordinate
        ret.x() = -(delta.dot(cameraLeft)   / pixelDistance);
        ret.y() = -(delta.dot(cameraUp) / pixelDistance);

        return ret;
    }

    bool onEdge(Vector2d point) {
        return  (abs(point.x()) < 0.00001 || abs(point.x()-resolution) < 0.00001) && (point.y() > -1 && point.y() < resolution+1)  ||
                (abs(point.y()) < 0.00001 || abs(point.y()-resolution) < 0.00001) && (point.x() > -1 && point.x() < resolution+1);
    }

    // returns the image coordinates of the line segment projected onto the imageplane
    // return value is x1, y1, x2, y2
    inline Eigen::Vector4d projectLine(Vector3d _origin, Vector3d _direction) {

        Eigen::ParametrizedLine<double,3> line = Eigen::ParametrizedLine<double,3> (_origin, _direction);

        // construct planes
        Eigen::Hyperplane<double,3> leftPlane = Eigen::Hyperplane<double,3>(
                topLeftPosition.cross(ray(Vector2d(0, resolution))), origin);
        Eigen::Hyperplane<double,3> topPlane = Eigen::Hyperplane<double,3>(
                topLeftPosition.cross(ray(Vector2d(resolution, 0))), origin);
        Eigen::Hyperplane<double,3> bottomPlane = Eigen::Hyperplane<double,3>(
                ray(Vector2d(resolution,resolution)).cross(ray(Vector2d(0, resolution))), origin);
        Eigen::Hyperplane<double,3> rightPlane = Eigen::Hyperplane<double,3>(
                ray(Vector2d(resolution,resolution)).cross(ray(Vector2d(resolution, 0))), origin);

        // intersections are correct!
        Vector3d left =     line.intersectionPoint(leftPlane);
        Vector3d top =      line.intersectionPoint(topPlane);
        Vector3d right =    line.intersectionPoint(rightPlane);
        Vector3d bottom =   line.intersectionPoint(bottomPlane);


        // construct projections of line - plane crossings
        Vector2d points[4];
        points[0] =  projectPoint(line.intersectionPoint(leftPlane));
        points[1] =  projectPoint(line.intersectionPoint(topPlane));
        points[2] =  projectPoint(line.intersectionPoint(rightPlane));
        points[3] =  projectPoint(line.intersectionPoint(bottomPlane));

        Vector2d buffer[4];
        int count = 0;

        for (int i = 0; i<4; i++) {
            if (onEdge(points[i])) {
                buffer[count] = points[i];
                count++;
            }
        }

        // would be strange if we found more than 2 valid points
        assert(count <= 2);

        Eigen::Vector4d res;
        if (count == 2)
            res = Eigen::Vector4d(buffer[0].x(),buffer[0].y(),buffer[1].x(), buffer[1].y());
        if (count <2)
            res = Eigen::Vector4d(0,0,resolution,resolution);
        return res;
    }
};

double intersect(const Vector3d& originA, const Vector3d& directionA, const Vector3d& originB, const Vector3d& directionB, Vector3d& point) {

    assert(abs(directionA.norm()-1)<0.00001);
    assert(abs(directionB.norm()-1)<0.00001);

    // perpendicular direction
    Vector3d perp = directionA.cross(directionB);

    // lines should not be parallel!
    assert(perp.norm() > 0.1);

    // normal for the planes
    const Vector3d normalA = directionA.cross(perp).normalized();
    const Vector3d normalB = directionB.cross(perp).normalized();

    // construct lines
    Eigen::ParametrizedLine<double,3> lineA = Eigen::ParametrizedLine<double,3> (originA, directionA);
    Eigen::ParametrizedLine<double,3> lineB = Eigen::ParametrizedLine<double,3> (originB, directionB);

    // construct planes
    Eigen::Hyperplane<double,3> planeA = Eigen::Hyperplane<double,3>(normalA, originA);
    Eigen::Hyperplane<double,3> planeB = Eigen::Hyperplane<double,3>(normalB, originB);

    // intersect
    Vector3d pointA = lineA.intersectionPoint(planeB);
    Vector3d pointB = lineB.intersectionPoint(planeA);

    point = (pointA + pointB) / 2;
    return (pointA-pointB).norm();
}


#endif //ARTERY_VISION_CAMERA_H
