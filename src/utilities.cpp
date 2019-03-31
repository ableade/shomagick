#include "utilities.h"

using cv::Point3d;
using cv::Vec3d;

ShoRowVector3d convertPointToRowVector(Point3d p) {
    return { p.x, p.y, p.z };
}

ShoRowVector3d convertVecToRowVector(Vec3d p) {
    return { p(0), p(1), p(2) };
}

ShoColumnVector3d convertVecToColumnVector(Vec3d p) {
    return { p(0), p(1), p(2) };
}
ShoRowVector4d convertColumnVecToRowVector(ShoColumnVector3d a) {
    return { a(0,0), a(1,0), a(2,0) };
}
