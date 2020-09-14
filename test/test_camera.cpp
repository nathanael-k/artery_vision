//
// Created by nate on 14.09.20.
//

#include "Base.h"

#include "../include/camera.h"

void testCamera()
{
    Camera basic = Camera("../../test/data/basic", 0);

    ASSERT_EQUAL_VEC( basic.cameraUp, Vector3d(0,0,1));
    ASSERT_EQUAL_VEC( basic.cameraLeft, Vector3d(0,1,0));

    ASSERT_CLOSE( (basic.ray(0,0) + basic.ray(basic.resolution,basic.resolution)).norm(), 2);
    ASSERT_CLOSE( basic.ray(basic.resolution/2,basic.resolution/2).x(), basic.direction.x());
}

void testIntersection()
{
    // same origin
    Vector3d point;
    double res = intersect(Vector3d(1,2,3), Vector3d(1,0,0),
                           Vector3d(1,2,3), Vector3d(0,1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(1,2,3));
    ASSERT_CLOSE( res, 0);

    // origin is closest point
    res = intersect( Vector3d(0,0,0), Vector3d(0,1,0),
                     Vector3d(1,0,0), Vector3d(0,0,1), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0.5,0,0));
    ASSERT_CLOSE( res, 1);

    // crossing lines
    res = intersect( Vector3d(1,0,0), Vector3d(-1,0,0),
                     Vector3d(0,1,0), Vector3d(0,-1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0,0,0));
    ASSERT_CLOSE( res, 0);

    // passing lines
    res = intersect( Vector3d(1,0,1), Vector3d(-1,0,0),
                     Vector3d(0,1,0), Vector3d(0,-1,0), point);
    ASSERT_CLOSE_VEC(point, Vector3d(0,0,0.5));
    ASSERT_CLOSE( res, 1);
}

int main(int, char**)
{
    testCamera();
    testIntersection();
}

