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

class Camera {

public:
    std::string name;

    // in m
    Vector3d origin;

    Vector3d direction;
    Vector3d cameraUp;
    Vector3d cameraLeft;

    Vector3d topLeftDirection;

    // in m
    double sensorSize = 0;
    double focalLength = 0;

    // pixels
    double resolution = 0;

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
                    pixelDistance = (sensorSize / resolution) * focalLength;

                    topLeftDirection = direction + (cameraUp + cameraLeft) * pixelDistance * (resolution / 2);

                    return;
                }
            }
        }

        std::cerr << "Unable to find cam " << index << " in file " << metaPath;
        exit(1);
    }

    // origin is top left, x is right, y is down (openCV convention)
    inline Vector3d ray(int x, int y) {
        return (topLeftDirection - cameraLeft * x * pixelDistance - cameraUp * y * pixelDistance).normalized();
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
