//
// Created by nate on 14.09.20.
//

#include "Base.h"

#include "../include/camera.h"

void testCamera()
{
    Camera basic = Camera("../../test/data/basic", 0);

    ASSERT_EQUAL_VEC( basic.cameraUp,   Vector3d(0,0,1));
    ASSERT_EQUAL_VEC( basic.cameraLeft, Vector3d(0,1,0));

    ASSERT_CLOSE_VEC( basic.ray(Vector2d(basic.resolution/2,basic.resolution/2)), basic.direction*basic.focalLength);
    ASSERT_CLOSE_VEC( (basic.ray(Vector2d(0,0)) + basic.ray(Vector2d(basic.resolution,basic.resolution))) , basic.direction*basic.focalLength*2 );


    Vector2d imageCoordinate = basic.projectPoint(basic.origin + basic.direction);
    ASSERT_CLOSE(imageCoordinate.x(), basic.resolution / 2);

    imageCoordinate = basic.projectPoint(basic.origin + basic.topLeftPosition);
    ASSERT_CLOSE(imageCoordinate.x(), 0);

    imageCoordinate = basic.projectPoint(Vector3d(1,0,0));
    ASSERT_CLOSE(imageCoordinate.x(), 512);
}

void testIntersection()
{
    // same origin
    Vector3d point;
    double res = Camera::intersect(Vector3d(1,2,3), Vector3d(1,0,0),
                           Vector3d(1,2,3), Vector3d(0,1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(1,2,3));
    ASSERT_CLOSE( res, 0);

    // origin is closest point
    res = Camera::intersect( Vector3d(0,0,0), Vector3d(0,1,0),
                     Vector3d(1,0,0), Vector3d(0,0,1), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0.5,0,0));
    ASSERT_CLOSE( res, 1);

    // crossing lines
    res = Camera::intersect( Vector3d(1,0,0), Vector3d(-1,0,0),
                     Vector3d(0,1,0), Vector3d(0,-1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0,0,0));
    ASSERT_CLOSE( res, 0);

    // passing lines
    res = Camera::intersect( Vector3d(1,0,1), Vector3d(-1,0,0),
                     Vector3d(0,1,0), Vector3d(0,-1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0,0,0.5));
    ASSERT_CLOSE( res, 1);
}

void testProjection()
{
    Camera well = Camera("../../test/data/welldefined", 0);
    Vector2d coordinate = well.projectPoint(Vector3d(0,1,1));

    Vector2d initial(0,0);
    Vector3d point = well.origin + well.ray(initial)*10;
    Vector2d selftest = well.projectPoint(point);
    ASSERT_CLOSE_VEC(initial, selftest);

    initial.x() = well.resolution/2;
    point = well.origin + well.ray(initial)*10;
    selftest = well.projectPoint(point);
    ASSERT_CLOSE_VEC(initial, selftest);

    initial.y() = well.resolution/2;
    point = well.origin + well.ray(initial)*10;
    selftest = well.projectPoint(point);
    ASSERT_CLOSE_VEC(initial, selftest);

    selftest = well.projectPoint(well.origin + well.ray(Vector2d(0,0)));

    // cams at 90°, looking at same 0,0,0, direction of one should cross the other in the middle
    Camera left     = Camera("../../test/data/projection", 0);
    Camera right    = Camera("../../test/data/projection", 1);

    Eigen::Vector4d result = left.projectLine(right.origin, right.direction);

}

int main(int, char**)
{
    testCamera();
    testIntersection();
    testProjection();
}

