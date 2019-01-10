/*
This code is taken from OpenSFM see License at

*/
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/StdVector>
#include "bootstrap.h"

namespace csfm
{

typedef std::vector<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4> > > vector_mat34;

enum {
  TRIANGULATION_OK = 0,
  TRIANGULATION_SMALL_ANGLE,
  TRIANGULATION_BEHIND_CAMERA,
  TRIANGULATION_BAD_REPROJECTION
};


double AngleBetweenVectors(const Eigen::Vector3d &u,
                           const Eigen::Vector3d &v);

bool TriangulateReturn(int error);


Eigen::Vector4d TriangulateBearingsDLTSolve(
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &bs,
    const vector_mat34 &Rts);

bool TriangulateBearingsDLT(
                                  double threshold,
                                  double min_angle);


// Point minimizing the squared distance to all rays
// Closed for solution from
//   Srikumar Ramalingam, Suresh K. Lodha and Peter Sturm
//   "A generic structure-from-motion framework"
//   CVIU 2006
Eigen::Vector3d TriangulateBearingsMidpointSolve(
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &os,
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &bs);

typedef double Radians;

bool TriangulateBearingsMidpoint(
    const std::vector<Eigen::Vector3d>& os_list,
    const std::vector<Eigen::Vector3d>& bs_list,
    Eigen::Vector3d& result,
    double treshold = 0.006,
    Radians min_angle = 1.0 * M_PI / 180
);

} //namespace csfm
