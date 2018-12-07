/*
This code is taken from OpenSFM see License at

*/
#include <iostream>
#include <fstream>
#include <string>
#include "types.h"
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/StdVector>


namespace csfm {


typedef std::vector<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4> > > vector_mat34;

enum {
  TRIANGULATION_OK = 0,
  TRIANGULATION_SMALL_ANGLE,
  TRIANGULATION_BEHIND_CAMERA,
  TRIANGULATION_BAD_REPROJECTION
};


double AngleBetweenVectors(const Eigen::Vector3d &u,
                           const Eigen::Vector3d &v);

bp::object TriangulateReturn(int error, bp::object value);


Eigen::Vector4d TriangulateBearingsDLTSolve(
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &bs,
    const vector_mat34 &Rts);

bp::object TriangulateBearingsDLT(const bp::list &Rts_list,
                                  const bp::list &bs_list,
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


bp::object TriangulateBearingsMidpoint(const bp::list &os_list,
                                       const bp::list &bs_list,
                                       const bp::list &threshold_list,
                                       double min_angle);


}

