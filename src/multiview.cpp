/*
This code is taken from OpenSFM see License at

*/
#include "multiview.h"
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "transformations.h"
#include "allclose.h"
#include "utilities.h"
#include <opencv2/core/eigen.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac/Ransac.hpp>

using cv::Mat;
using cv::Scalar;
using std::vector;
using cv::Point3d;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using opengv::absolute_pose::CentralAbsoluteAdapter;
using opengv::sac::Ransac;
using opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using opengv::transformation_t;

namespace csfm
{
  double AngleBetweenVectors(const Eigen::Vector3d &u,
                            const Eigen::Vector3d &v)
  {
    double c = (u.dot(v)) / sqrt(u.dot(u) * v.dot(v));
    if (c >= 1.0)
      return 0.0;
    else
      return acos(c);
  }

  // Point minimizing the squared distance to all rays
// Closed for solution from
//   Srikumar Ramalingam, Suresh K. Lodha and Peter Sturm
//   "A generic structure-from-motion framework"
//   CVIU 2006
Eigen::Vector3d TriangulateBearingsMidpointSolve(const Eigen::Matrix<double, 3, Eigen::Dynamic> &os,
                                                 const Eigen::Matrix<double, 3, Eigen::Dynamic> &bs)
{
  int nviews = bs.cols();
  assert(nviews == os.cols());
  assert(nviews >= 2);

  Eigen::Matrix3d BBt;
  Eigen::Vector3d BBtA, A;
  BBt.setZero();
  BBtA.setZero();
  A.setZero();
  for (int i = 0; i < nviews; ++i)
  {
    BBt += bs.col(i) * bs.col(i).transpose();
    BBtA += bs.col(i) * bs.col(i).transpose() * os.col(i);
    A += os.col(i);
  }
  Eigen::Matrix3d Cinv = (nviews * Eigen::Matrix3d::Identity() - BBt).inverse();

  return (Eigen::Matrix3d::Identity() + BBt * Cinv) * A / nviews - Cinv * BBtA;
}

bool TriangulateBearingsMidpoint(
    const std::vector<Eigen::Vector3d> &os_list,
    const std::vector<Eigen::Vector3d> &bs_list,
    Eigen::Vector3d &result,
    double threshold,
    Radians min_angle
)
{
  int n = os_list.size();

  // Build Eigen matrices
  Eigen::Matrix<double, 3, Eigen::Dynamic> os(3, n);
  Eigen::Matrix<double, 3, Eigen::Dynamic> bs(3, n);
  for (int i = 0; i < n; ++i)
  {
    auto o = os_list[i];
    auto b = bs_list[i];
    os.col(i) << o(0), o(1), o(2);
    bs.col(i) << b(0), b(1), b(2);
  }
  //std::cout << "Triangulating bearings midpoint for os " << os << std::endl;
  //std::cout << "Triangulating bearings midpoint for bs " << bs << std::endl;
  // Check angle between rays
  bool angle_ok = false;
  for (int i = 0; i < n; ++i)
  {
    if (!angle_ok)
    {
      for (int j = 0; j < i; ++j)
      {
        Eigen::Vector3d a, b;
        a << bs(0, i), bs(1, i), bs(2, i);
        b << bs(0, j), bs(1, j), bs(2, j);
        double angle = AngleBetweenVectors(a, b);
        if (angle >= min_angle)
        {
          angle_ok = true;
        }
      }
    }
  }
  if (!angle_ok)
  {
    //std::cout << "Angle is not OK at all" << std::endl;
    return false;
  }

  //std::cout << "Angle was Ok" << std::endl;

  // Triangulate
  result = TriangulateBearingsMidpointSolve(os, bs);
  //std::cout << "X is " << result << std::endl;

  // Check reprojection error
  for (int i = 0; i < n; ++i)
  {
    Eigen::Vector3d x_reproj = result - os.col(i);
    Eigen::Vector3d b = bs.col(i);

    double error = AngleBetweenVectors(x_reproj, b);
    //std::cout << "Error was " << error << std::endl;
    if (error > threshold)
    {
      //std::cout << "Error was greater than treshhold" << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace csfm

/*
bp::object TriangulateReturn(int error, bp::object value)
{
  bp::list retn;
  retn.append(int(error));
  retn.append(value);
  return retn;
}

Eigen::Vector4d TriangulateBearingsDLTSolve(
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &bs,
    const vector_mat34 &Rts)
{
  int nviews = bs.cols();
  assert(nviews == Rts.size());

  Eigen::MatrixXd A(2 * nviews, 4);
  for (int i = 0; i < nviews; i++)
  {
    A.row(2 * i) = bs(0, i) * Rts[i].row(2) - bs(2, i) * Rts[i].row(0);
    A.row(2 * i + 1) = bs(1, i) * Rts[i].row(2) - bs(2, i) * Rts[i].row(1);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> mySVD(A, Eigen::ComputeFullV);
  Eigen::Vector4d worldPoint;
  worldPoint[0] = mySVD.matrixV()(0, 3);
  worldPoint[1] = mySVD.matrixV()(1, 3);
  worldPoint[2] = mySVD.matrixV()(2, 3);
  worldPoint[3] = mySVD.matrixV()(3, 3);

  return worldPoint;
}

bp::object TriangulateBearingsDLT(const bp::list &Rts_list,
                                  const bp::list &bs_list,
                                  double threshold,
                                  double min_angle)
{

  int n = bp::len(Rts_list);
  vector_mat34 Rts;
  Eigen::Matrix<double, 3, Eigen::Dynamic> bs(3, n);
  Eigen::MatrixXd vs(3, n);
  bool angle_ok = false;
  for (int i = 0; i < n; ++i)
  {
    bp::object oRt = Rts_list[i];
    bp::object ob = bs_list[i];

    PyArrayContiguousView<double> Rt_array(oRt);
    PyArrayContiguousView<double> b_array(ob);

    Eigen::Map<const Eigen::MatrixXd> Rt(Rt_array.data(), 4, 3);
    Eigen::Map<const Eigen::MatrixXd> b(b_array.data(), 3, 1);

    Rts.push_back(Rt.transpose());
    bs.col(i) = b.col(0);

    // Check angle between rays
    if (!angle_ok)
    {
      Eigen::Vector3d xh;
      xh << b(0, 0), b(1, 0), b(2, 0);
      Eigen::Vector3d v = Rt.block<3, 3>(0, 0).transpose().inverse() * xh;
      vs.col(i) << v(0), v(1), v(2);

      for (int j = 0; j < i; ++j)
      {
        Eigen::Vector3d a, b;
        a << vs(0, i), vs(1, i), vs(2, i);
        b << vs(0, j), vs(1, j), vs(2, j);
        double angle = AngleBetweenVectors(a, b);
        if (angle >= min_angle)
        {
          angle_ok = true;
        }
      }
    }
  }

  if (!angle_ok)
  {
    return TriangulateReturn(TRIANGULATION_SMALL_ANGLE, bp::object());
  }

  Eigen::Vector4d X = TriangulateBearingsDLTSolve(bs, Rts);
  X /= X(3);

  for (int i = 0; i < n; ++i)
  {
    Eigen::Vector3d x_reproj = Rts[i] * X;
    Eigen::Vector3d b;
    b << bs(0, i), bs(1, i), bs(2, i);

    double error = AngleBetweenVectors(x_reproj, b);
    if (error > threshold)
    {
      return TriangulateReturn(TRIANGULATION_BAD_REPROJECTION, bp::object());
    }
  }

  return TriangulateReturn(TRIANGULATION_OK,
                           bpn_array_from_data(X.data(), 3));
}

*/

/*
Estimates a plane from on-plane points and vectors.
See https://raw.githubusercontent.com/mapillary/OpenSfM/master/opensfm/reconstruction.py
*/
cv::Vec4d fitPlane(std::vector<cv::Vec3d> points, std::vector<cv::Vec3d> vectors, std::vector<cv::Vec3d> verticals)
{
    Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
    Mat pointsMat(points);
    meanStdDev(pointsMat.reshape(1,1), mean, stddev, cv::Mat());
    //std::cout << "Std of points was " << stddev[0] << "\n";
    const auto s = 1.0 / std::max(1e-8, stddev[0]);
    //std::cout << "S was " << s << "\n";
    Mat x;
    Mat vectorsMat(vectors);
    cv::convertPointsToHomogeneous(s* pointsMat, x);
    //std::cout << "Homogenous x is " << x << "\n";
    Mat a;
    //TODO investigate if this condition is needed
    if (!vectors.empty()) {
        Mat homogenousVec;
        convertVectorToHomogeneous(vectorsMat, homogenousVec);
        cv::vconcat(x, homogenousVec, a);
    }
    else {
        a = x;
    }
    a = a.reshape(1);
    auto[o, p] = nullSpace(a);
    std::cout << "P returned from nullspace is " << p << "\n";
    p.at<double>(0, 3) /= s;
    const auto pRange = p.colRange(0, 3);
    //std::cout << "Type of prange is " << pRange.type() << "\n";
    //std::cout << "Size of prange is " << pRange.size() << "\n";
    if (allClose(p.colRange(0, 3), Mat::zeros({ 3,1 }, CV_64F))) {
        return { 0.0, 0.0, 1.0, 0 };
    }

    if (!verticals.empty()) {
        auto d = 0;
        for (const auto& vertical : verticals) {
            std::cout << "Vertical is now " << vertical << "\n\n";
            const auto verticalMat = Mat(verticals).reshape(1, 1);
            std::cout << "Size of vertical mat  is now " << verticalMat.size() << "\n\n";
            auto pRange = p.colRange(0, 3);
            d+=  pRange.dot(verticalMat);
            std::cout << "D is now " << d << "\n\n";
            p *= sgn(d);
        }
    }
    return p;
}

void convertVectorToHomogeneous(cv::InputArray _src  , cv::OutputArray _dst) {
    Mat src = _src.getMat();
    if (!src.isContinuous())
        src = src.clone();
    int i, npoints = src.checkVector(2), depth = src.depth(), cn = 2;
    if (npoints < 0)
    {
        npoints = src.checkVector(3);
        CV_Assert(npoints >= 0);
        cn = 3;
    }
    CV_Assert(npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth, cn + 1);
    _dst.create(npoints, 1, dtype);
    Mat dst = _dst.getMat();
    if (!dst.isContinuous())
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert(dst.isContinuous());

    if (depth == CV_32S)


    {
        if (cn == 2)
        {
            const cv::Point2i* sptr = src.ptr<cv::Point2i>();
            cv::Point3i* dptr = dst.ptr<cv::Point3i>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Point3i(sptr[i].x, sptr[i].y, 0);
        }
        else
        {
            const cv::Point3i* sptr = src.ptr<cv::Point3i>();
            cv::Vec4i* dptr = dst.ptr<cv::Vec4i>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Vec4i(sptr[i].x, sptr[i].y, sptr[i].z, 0);
        }
    }
    else if (depth == CV_32F)
    {
        if (cn == 2)
        {
            const cv::Point2f* sptr = src.ptr<cv::Point2f>();
            cv::Point3f* dptr = dst.ptr<cv::Point3f>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Point3f(sptr[i].x, sptr[i].y, 0.f);
        }
        else
        {
            const cv::Point3f* sptr = src.ptr<cv::Point3f>();
            cv::Vec4f* dptr = dst.ptr<cv::Vec4f>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Vec4f(sptr[i].x, sptr[i].y, sptr[i].z, 0.f);
        }
    }
    else if (depth == CV_64F)
    {
        if (cn == 2)
        {
            const cv::Point2d* sptr = src.ptr<cv::Point2d>();
            cv::Point3d* dptr = dst.ptr<cv::Point3d>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Point3d(sptr[i].x, sptr[i].y, 0.);
        }
        else
        {
            const cv::Point3d* sptr = src.ptr<cv::Point3d>();
            cv::Vec4d* dptr = dst.ptr<cv::Vec4d>();
            for (i = 0; i < npoints; i++)
                dptr[i] = cv::Vec4d(sptr[i].x, sptr[i].y, sptr[i].z, 0.);
        }
    }
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}

std::tuple<cv::Mat, cv::Mat> nullSpace(cv::Mat a)
{
    auto svd = cv::SVD();
    Mat u, s, vh;
    std::cout << "A is " << a << "\n\n";
    svd.compute(a, u, s, vh, cv::SVD::FULL_UV);
    std::cout << " u returned from svd is " << u << "\n\n";
    std::cout << " S returned from svd is " << s << "\n\n";
    std::cout << " VH returned from svd is " << vh << "\n\n";
    return std::make_tuple(s.row(u.rows - 1), vh.row(vh.rows - 1));
}
Matrix3d calculateHorizontalPlanePosition(cv::Mat p)
{
    const auto v0ColRange = p.colRange(0, 3);
    const auto v0 = Point3d(v0ColRange);
    
    const Point3d v1{ 0.0, 0.1, 1.0 };

    const auto angle = calculateAngleBetweenVectors(v0, v1);
    const auto axis = v0.cross(v1);
    Vector3d eigenAxis;
    cv2eigen(Mat(axis), eigenAxis);
    const auto norm = eigenAxis.norm();

    if (norm > 0) {
        auto m = rotationMatrix(angle, eigenAxis, nullptr);
        Matrix3d rot;
        rot = m.block<3, 3>(0, 0);
        return rot;
    }
    else if (angle < 1.0) {
        return Matrix3d::Identity();
    }
    else if (angle > 3.0) {
        Vector3d diagonal{ 1, -1, -1 };
        return diagonal.asDiagonal().toDenseMatrix();
    }
    return Matrix3d();
}

std::vector<float> getStdByAxis(cv::InputArray data, int axis) {
    auto m = data.getMat().reshape(1);
    cv::Scalar mean, stddev;
    std::vector<float> stds;
    if (!axis) {
        for (int i = 0; i < m.cols; ++i) {
            cv::meanStdDev(m.col(i), mean, stddev);
            stds.push_back(stddev[0]);
        }
    }
    else {
        for (int i = 0; i < m.rows; ++i) {
            cv::meanStdDev(m.row(i), mean, stddev);
            stds.push_back(stddev[0]);
        }
    }
    return stds;
}

std::vector<float> getMeanByAxis(cv::InputArray data, int axis)
{
    auto m = data.getMat().reshape(1);
    cv::Scalar mean, stddev;
    std::vector<float> means;
    if (!axis) {
        for (int i = 0; i < m.cols; ++i) {
            cv::meanStdDev(m.col(i), mean, stddev);
            means.push_back(mean[0]);
        }
    }
    else {
        for (int i = 0; i < m.rows; ++i) {
            cv::meanStdDev(m.row(i), mean, stddev);
            means.push_back(mean[0]);
        }
    }
    return means;
}

transformation_t absolutePoseRansac(opengv::bearingVectors_t bearings, opengv::points_t points, double threshold, int iterations, double probability)
{
    // create the central adapter
    CentralAbsoluteAdapter adapter(
        bearings, points);
    // create a Ransac object
    Ransac<AbsolutePoseSacProblem> ransac;
    // create an AbsolutePoseSacProblem
    // (algorithm is selectable: KNEIP, GAO, or EPNP)
    std::shared_ptr<AbsolutePoseSacProblem>
        absposeproblem_ptr(
            new AbsolutePoseSacProblem(adapter, AbsolutePoseSacProblem::KNEIP));
    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = threshold;
    ransac.max_iterations_ = iterations;
    ransac.probability_ = probability;
    ransac.computeModel();
    // get the result
    transformation_t best_transformation = ransac.model_coefficients_;

    return best_transformation;
}
