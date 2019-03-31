#pragma once
#include <opencv2/core.hpp>
#include <Eigen/core>

#define EPSILON 8.8817841970012523e-016 /* 4.0 * DBL_EPSILON */

cv::Mat_<double> vectorNorm(cv::InputArray src, int axis);

double calculateAngleBetweenVectors(const cv::Point3d& v1, const cv::Point3d& v2, bool directed = true);

cv::Mat_<double> calculateAngleBetweenVectors(const std::vector<cv::Vec4d> v1,
    const std::vector<double> v2, int axis = 0, bool directed = true);

cv::Mat_<double> calculateAngleBetweenVectors(const std::vector<cv::Vec3d> v1,
    const std::vector<cv::Vec3d> v2, int axis = 1, bool directed = true);

Eigen::Matrix4d rotationMatrix(double angle, Eigen::Vector3d direction, double *p);

/*
Quaternion to rotation matrix.
*/
int quaternion_matrix(
    double *quat,    /* double[4]  */
    double *matrix);

/*
Tridiagonal matrix from symmetric 4x4 matrix using Housholder reduction.
The input matrix is altered.
*/
int tridiagonalize_symmetric_44(
    double *matrix,      /* double[16] */
    double *diagonal,    /* double[4] */
    double *subdiagonal);

/*
Return largest eigenvalue of symmetric tridiagonal matrix.
Matrix Algorithms: Volume II: Eigensystems. By GW Stewart.
Chapter 3. page 197.
*/
double max_eigenvalue_of_tridiag_44(
    double *diagonal,    /* double[4] */
    double *subdiagonal); /* double[3] */

/*
Eigenvector of symmetric tridiagonal 4x4 matrix using Cramer's rule.
*/
int eigenvector_of_symmetric_44(
    double *matrix, /* double[16] */
    double *vector, /* double[4]  */
    double *buffer); /* double[12] */

Eigen::Matrix4d superImpositionMatrix(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b,
    bool scaling = false);