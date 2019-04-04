#include "transformations.h"
#include "utilities.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"

using cv::Point;
using cv::Mat_;
using cv::Mat;
using cv::InputArray;
using cv::Scalar;
using cv::REDUCE_SUM;
using cv::estimateAffinePartial2D;
using std::cout;
using Eigen::Matrix;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Dynamic;
using Eigen::RowMajor;

Mat_<double> vectorNorm(InputArray src, int axis)
{
    Mat dataSquared = src.getMat().reshape(1);
    dataSquared = dataSquared.mul(dataSquared);
    reduce(dataSquared, dataSquared, axis, REDUCE_SUM);
    cv::sqrt(dataSquared, dataSquared);
    return dataSquared;
}

double calculateAngleBetweenVectors(const cv::Point3d& v1, const cv::Point3d& v2, bool directed)
{
    double cosAngle = v1.dot(v2) / (cv::norm(v1) * cv::norm(v2));
    cout << "Cos angle is " << cosAngle << "\n\n";
    if (cosAngle > 1.0)
        return 0.0;
    else if (cosAngle < -1.0)
        return CV_PI;
    if (!directed)
        cosAngle = std::abs(cosAngle);
    return std::acos(cosAngle);
}

Mat_<double> calculateAngleBetweenVectors(const std::vector<cv::Vec4d> v1,
    const std::vector<double> v2, int axis, bool directed)
{
    CV_Assert(v2.size() == v1.size());
    auto matPoints = Mat(v1).reshape(1);
    auto v2Mat = Mat(v2).reshape(1);

    while (v2Mat.total() < matPoints.rows * matPoints.cols) {
        v2Mat.push_back(Mat(v2).reshape(1));
    }
    v2Mat = v2Mat.reshape(1, { matPoints.cols, matPoints.rows });

    rotate(v2Mat, v2Mat, cv::ROTATE_90_CLOCKWISE);

    Mat_<double> dot = matPoints.mul(v2Mat);
    reduce(dot, dot, axis, REDUCE_SUM);

    auto v1Norm = vectorNorm(matPoints, axis);
    auto v2Norm = vectorNorm(Mat(v2), axis);;

    multiplyMat_ByScalar(v1Norm, v2Norm(0, 0));
    divide(dot, v1Norm, dot);
    if (!directed) {
        dot = abs(dot);
    }

    for (auto& element : dot) {
        element = std::acos(element);
    }
    return dot;
}

Mat_<double> calculateAngleBetweenVectors(const std::vector<cv::Vec3d> v1,
    const std::vector<cv::Vec3d> v2, int axis, bool directed) {

    CV_Assert(v1.size() == v2.size());
    auto matV1 = Mat(v1);
    auto matV2 = Mat(v2);
    const auto prod = Mat(matV1.mul(matV2));
    Mat dot;
    reduce(prod.reshape(1), dot, axis, REDUCE_SUM);

    Mat v1Norm = vectorNorm(matV1, axis);
    Mat v2Norm = vectorNorm(matV2, axis);

    cout << "V1 norm is " << v1Norm << "\n\n";
    cout << "V2 norm is " << v2Norm << "\n\n";

    Mat prodNorm = v1Norm.mul(v2Norm);

    divide(dot, prodNorm, dot);

    if (!directed) {
        dot = abs(dot);
    }

    for (auto i = 0; i < dot.rows; ++i) {
        dot.at<double>(i, 0) = std::acos(dot.at<double>(i, 0));
    }
    return dot;
}

Matrix4d rotationMatrix(double angle, Vector3d direction, double *p) {
    Matrix<double, 4, 4, Eigen::RowMajor> out;
    double *d = direction.data();
    double dx = d[0];
    double dy = d[1];
    double dz = d[2];
    double sa = sin(angle);
    double ca = cos(angle);
    double ca1 = 1 - ca;
    double s, t;

    t = sqrt(dx*dx + dy * dy + dz * dz);
    if (t < EPSILON) {
        cout << "Got an invalid direction vector" << "\n\n";
        exit(1);
    }
    dx /= t;
    dy /= t;
    dz /= t;

    double *M = out.data();

    M[0] = ca + dx * dx * ca1;
    M[5] = ca + dy * dy * ca1;
    M[10] = ca + dz * dz * ca1;

    s = dz * sa;
    t = dx * dy * ca1;
    M[1] = t - s;
    M[4] = t + s;

    s = dy * sa;
    t = dx * dz * ca1;
    M[2] = t + s;
    M[8] = t - s;

    s = dx * sa;
    t = dy * dz * ca1;
    M[6] = t - s;
    M[9] = t + s;

    M[12] = M[13] = M[14] = 0.0;
    M[15] = 1.0;

    if (p != nullptr) {
        cout << "Point was not null" << "\n\n";
        M[3] = p[0] - (M[0] * p[0] + M[1] * p[1] + M[2] * p[2]);
        M[7] = p[1] - (M[4] * p[0] + M[5] * p[1] + M[6] * p[2]);
        M[11] = p[2] - (M[8] * p[0] + M[9] * p[1] + M[10] * p[2]);
    }
    else {
        M[3] = M[7] = M[11] = 0.0;
    }
    return out;
}

/*
Tridiagonal matrix from symmetric 4x4 matrix using Housholder reduction.
The input matrix is altered.
*/
int tridiagonalize_symmetric_44(
    double *matrix,      /* double[16] */
    double *diagonal,    /* double[4] */
    double *subdiagonal) /* double[3] */
{
    double t, n, g, h, v0, v1, v2;
    double *u;
    double *M = matrix;

    u = &M[1];
    t = u[1] * u[1] + u[2] * u[2];
    n = sqrt(u[0] * u[0] + t);
    if (n > EPSILON) {
        if (u[0] < 0.0)
            n = -n;
        u[0] += n;
        h = (u[0] * u[0] + t) / 2.0;
        v0 = M[5] * u[0] + M[6] * u[1] + M[7] * u[2];
        v1 = M[6] * u[0] + M[10] * u[1] + M[11] * u[2];
        v2 = M[7] * u[0] + M[11] * u[1] + M[15] * u[2];
        v0 /= h;
        v1 /= h;
        v2 /= h;
        g = (u[0] * v0 + u[1] * v1 + u[2] * v2) / (2.0 * h);
        v0 -= g * u[0];
        v1 -= g * u[1];
        v2 -= g * u[2];
        M[5] -= 2.0*v0*u[0];
        M[10] -= 2.0*v1*u[1];
        M[15] -= 2.0*v2*u[2];
        M[6] -= v1 * u[0] + v0 * u[1];
        M[7] -= v2 * u[0] + v0 * u[2];
        M[11] -= v2 * u[1] + v1 * u[2];
        M[1] = -n;
    }

    u = &M[6];
    t = u[1] * u[1];
    n = sqrt(u[0] * u[0] + t);
    if (n > EPSILON) {
        if (u[0] < 0.0)
            n = -n;
        u[0] += n;
        h = (u[0] * u[0] + t) / 2.0;
        v0 = M[10] * u[0] + M[11] * u[1];
        v1 = M[11] * u[0] + M[15] * u[1];
        v0 /= h;
        v1 /= h;
        g = (u[0] * v0 + u[1] * v1) / (2.0 * h);
        v0 -= g * u[0];
        v1 -= g * u[1];
        M[10] -= 2.0*v0*u[0];
        M[15] -= 2.0*v1*u[1];
        M[11] -= v1 * u[0] + v0 * u[1];
        M[6] = -n;
    }

    diagonal[0] = M[0];
    diagonal[1] = M[5];
    diagonal[2] = M[10];
    diagonal[3] = M[15];
    subdiagonal[0] = M[1];
    subdiagonal[1] = M[6];
    subdiagonal[2] = M[11];

    return 0;
}

Matrix4d superImpositionMatrix(Matrix<double, Dynamic, Dynamic, RowMajor> a,
    Matrix<double, Dynamic, Dynamic, RowMajor> b, bool scaling)
{
    Matrix<double, 4, 4, RowMajor> out;
    auto M = out.data();
    auto size = a.cols();
    double v0t[3], v1t[3];
    double t;
    for (auto j = 0; j < 3; ++j) {
        t = 0.0;
        for (auto i = 0; i < size; ++i) {
            t += a(j, i);
        }
        v0t[j] = t / (double)size;
    }

    for (auto j = 0; j < 3; ++j) {
        t = 0.0;
        for (auto i = 0; i < size; ++i) {
            t += b(j, i);
        }
        v1t[j] = t / (double)size;
    }

    double xx, yy, zz, xy, yz, zx, xz, yx, zy;
    xx = yy = zz = xy = yz = zx = xz = yx = zy = 0.0;

    double v0x, v0y, v0z;
    auto buffer = (double *)malloc(42 * sizeof(double));
    auto q = buffer;
    auto N = (buffer + 4);

    for (auto j = 0; j < size; ++j) {
        v0x = a(0, j) - v0t[0];
        v0y = a(1, j) - v0t[1];
        v0z = a(2, j) - v0t[2];

        t = b(0, j) - v1t[0];
        xx += v0x * t;
        yx += v0y * t;
        zx += v0z * t;
        t = b(1, j) - v1t[1];
        xy += v0x * t;
        yy += v0y * t;
        zy += v0z * t;
        t = b(2, j) - v1t[2];
        xz += v0x * t;
        yz += v0y * t;
        zz += v0z * t;

        N[0] = xx + yy + zz;
        N[5] = xx - yy - zz;
        N[10] = -xx + yy - zz;
        N[15] = -xx - yy + zz;
        N[1] = N[4] = yz - zy;
        N[2] = N[8] = zx - xz;
        N[3] = N[12] = xy - yx;
        N[6] = N[9] = xy + yx;
        N[7] = N[13] = zx + xz;
        N[11] = N[14] = yz + zy;

        double l;
        double *a = (buffer + 20);
        double *b = (buffer + 24);
        double *t = (buffer + 27);

        for (auto i = 0; i < 16; i++)
            M[i] = N[i];

        if (tridiagonalize_symmetric_44(M, a, b) != 0) {
            cout << "tridiagonalize_symmetric_44() failed\n";
            free(buffer);
            exit(1);
        }

        l = max_eigenvalue_of_tridiag_44(a, b);
        N[0] -= l;
        N[5] -= l;
        N[10] -= l;
        N[15] -= l;

        if (eigenvector_of_symmetric_44(N, q, t) != 0) {
            auto ev = eigenvector_of_symmetric_44(N, q, t);
            cout << " EV EAS " << ev << "\n\n";
            cout << "eigenvector_of_symmetric_44() failed\n";
            //free(buffer);
            //exit(1);
        }

        l = q[3];
        q[3] = q[2];
        q[2] = q[1];
        q[1] = q[0];
        q[0] = l;
    }

    if (quaternion_matrix(q, M) != 0) {
        cout << "quaternion_matrix() failed\n";
        free(buffer);
        exit(1);
    }

    if (scaling) {
        /* scaling factor = sqrt(sum(v1) / sum(v0) */
        double t, dt;
        double v0s = 0.0;
        double v1s = 0.0;

        for (auto j = 0; j < 3; ++j) {
            dt = v0t[j];
            for (auto i = 0; i < size; i++) {
                t = a(j, i) - dt;
                v0s += t * t;
            }
        }
        for (auto j = 0; j < 3; j++) {

            dt = v1t[j];
            for (auto i = 0; i < size; i++) {
                t = b(j, i) - dt;
                v1s += t * t;
            }
        }

        t = sqrt(v1s / v0s);
        M[0] *= t;
        M[1] *= t;
        M[2] *= t;
        M[4] *= t;
        M[5] *= t;
        M[6] *= t;
        M[8] *= t;
        M[9] *= t;
        M[10] *= t;
    }
    free(buffer);
    return out;
}

/*
Quaternion to rotation matrix.
*/
int quaternion_matrix(
    double *quat,    /* double[4]  */
    double *matrix)  /* double[16] */
{
    double *M = matrix;
    double *q = quat;
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

    if (n < EPSILON) {
        /* return identity matrix */
        memset(M, 0, 16 * sizeof(double));
        M[0] = M[5] = M[10] = M[15] = 1.0;
    }
    else {
        q[0] /= n;
        q[1] /= n;
        q[2] /= n;
        q[3] /= n;
        {
            double x2 = q[1] + q[1];
            double y2 = q[2] + q[2];
            double z2 = q[3] + q[3];
            {
                double xx2 = q[1] * x2;
                double yy2 = q[2] * y2;
                double zz2 = q[3] * z2;
                M[0] = 1.0 - yy2 - zz2;
                M[5] = 1.0 - xx2 - zz2;
                M[10] = 1.0 - xx2 - yy2;
            } {
                double yz2 = q[2] * z2;
                double wx2 = q[0] * x2;
                M[6] = yz2 - wx2;
                M[9] = yz2 + wx2;
            } {
                double xy2 = q[1] * y2;
                double wz2 = q[0] * z2;
                M[1] = xy2 - wz2;
                M[4] = xy2 + wz2;
            } {
                double xz2 = q[1] * z2;
                double wy2 = q[0] * y2;
                M[8] = xz2 - wy2;
                M[2] = xz2 + wy2;
            }
            M[3] = M[7] = M[11] = M[12] = M[13] = M[14] = 0.0;
            M[15] = 1.0;
        }
    }
    return 0;
}

/*
Return largest eigenvalue of symmetric tridiagonal matrix.
Matrix Algorithms: Volume II: Eigensystems. By GW Stewart.
Chapter 3. page 197.
*/
double max_eigenvalue_of_tridiag_44(
    double *diagonal,    /* double[4] */
    double *subdiagonal) /* double[3] */
{
    int count;
    double lower, upper, t0, t1, d, eps, eigenv;
    double *a = diagonal;
    double *b = subdiagonal;

    /* upper and lower bounds using Gerschgorin's theorem */
    t0 = fabs(b[0]);
    t1 = fabs(b[1]);
    lower = a[0] - t0;
    upper = a[0] + t0;
    d = a[1] - t0 - t1;
    lower = MIN(lower, d);
    d = a[1] + t0 + t1;
    upper = MAX(upper, d);
    t0 = fabs(b[2]);
    d = a[2] - t0 - t1;
    lower = MIN(lower, d);
    d = a[2] + t0 + t1;
    upper = MAX(upper, d);
    d = a[3] - t0;
    lower = MIN(lower, d);
    d = a[3] + t0;
    upper = MAX(upper, d);

    /* precision */
    /* eps = (4.0 * (fabs(lower) + fabs(upper))) * DBL_EPSILON; */
    eps = 1e-18;

    /* interval bisection until width is less than tolerance */
    while (fabs(upper - lower) > eps) {

        eigenv = (upper + lower) / 2.0;

        if ((eigenv == upper) || (eigenv == lower))
            return eigenv;

        /* counting pivots < 0 */
        d = a[0] - eigenv;
        count = (d < 0.0) ? 1 : 0;
        if (fabs(d) < eps)
            d = eps;
        d = a[1] - eigenv - b[0] * b[0] / d;
        if (d < 0.0)
            count++;
        if (fabs(d) < eps)
            d = eps;
        d = a[2] - eigenv - b[1] * b[1] / d;
        if (d < 0.0)
            count++;
        if (fabs(d) < eps)
            d = eps;
        d = a[3] - eigenv - b[2] * b[2] / d;
        if (d < 0.0)
            count++;

        if (count < 4)
            lower = eigenv;
        else
            upper = eigenv;
    }

    return (upper + lower) / 2.0;
}

/*
Eigenvector of symmetric tridiagonal 4x4 matrix using Cramer's rule.
*/
int eigenvector_of_symmetric_44(
    double *matrix, /* double[16] */
    double *vector, /* double[4]  */
    double *buffer) /* double[12] */
{
    double n, eps;
    double *M = matrix;
    double *v = vector;
    double *t = buffer;

    /* eps: minimum length of eigenvector to use */
    eps = (M[0] * M[5] * M[10] * M[15] - M[1] * M[1] * M[11] * M[11]) * 1e-6;
    eps *= eps;
    if (eps < EPSILON)
        eps = EPSILON;

    t[0] = M[10] * M[15];
    t[1] = M[11] * M[11];
    t[2] = M[6] * M[15];
    t[3] = M[11] * M[7];
    t[4] = M[6] * M[11];
    t[5] = M[10] * M[7];
    t[6] = M[2] * M[15];
    t[7] = M[11] * M[3];
    t[8] = M[2] * M[11];
    t[9] = M[10] * M[3];
    t[10] = M[2] * M[7];
    t[11] = M[6] * M[3];

    v[0] = t[1] * M[1] + t[6] * M[6] + t[9] * M[7];
    v[0] -= t[0] * M[1] + t[7] * M[6] + t[8] * M[7];
    v[1] = t[2] * M[1] + t[7] * M[5] + t[10] * M[7];
    v[1] -= t[3] * M[1] + t[6] * M[5] + t[11] * M[7];
    v[2] = t[5] * M[1] + t[8] * M[5] + t[11] * M[6];
    v[2] -= t[4] * M[1] + t[9] * M[5] + t[10] * M[6];
    v[3] = t[0] * M[5] + t[3] * M[6] + t[4] * M[7];
    v[3] -= t[1] * M[5] + t[2] * M[6] + t[5] * M[7];
    n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];

    if (n < eps) {
        v[0] = t[0] * M[0] + t[7] * M[2] + t[8] * M[3];
        v[0] -= t[1] * M[0] + t[6] * M[2] + t[9] * M[3];
        v[1] = t[3] * M[0] + t[6] * M[1] + t[11] * M[3];
        v[1] -= t[2] * M[0] + t[7] * M[1] + t[10] * M[3];
        v[2] = t[4] * M[0] + t[9] * M[1] + t[10] * M[2];
        v[2] -= t[5] * M[0] + t[8] * M[1] + t[11] * M[2];
        v[3] = t[1] * M[1] + t[2] * M[2] + t[5] * M[3];
        v[3] -= t[0] * M[1] + t[3] * M[2] + t[4] * M[3];
        n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    }

    if (n < eps) {
        t[0] = M[2] * M[7];
        t[1] = M[3] * M[6];
        t[2] = M[1] * M[7];
        t[3] = M[3] * M[5];
        t[4] = M[1] * M[6];
        t[5] = M[2] * M[5];
        t[6] = M[0] * M[7];
        t[7] = M[3] * M[1];
        t[8] = M[0] * M[6];
        t[9] = M[2] * M[1];
        t[10] = M[0] * M[5];
        t[11] = M[1] * M[1];

        v[0] = t[1] * M[3] + t[6] * M[11] + t[9] * M[15];
        v[0] -= t[0] * M[3] + t[7] * M[11] + t[8] * M[15];
        v[1] = t[2] * M[3] + t[7] * M[7] + t[10] * M[15];
        v[1] -= t[3] * M[3] + t[6] * M[7] + t[11] * M[15];
        v[2] = t[5] * M[3] + t[8] * M[7] + t[11] * M[11];
        v[2] -= t[4] * M[3] + t[9] * M[7] + t[10] * M[11];
        v[3] = t[0] * M[7] + t[3] * M[11] + t[4] * M[15];
        v[3] -= t[1] * M[7] + t[2] * M[11] + t[5] * M[15];
        n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    }

    if (n < eps) {
        v[0] = t[8] * M[11] + t[0] * M[2] + t[7] * M[10];
        v[0] -= t[6] * M[10] + t[9] * M[11] + t[1] * M[2];
        v[1] = t[6] * M[6] + t[11] * M[11] + t[3] * M[2];
        v[1] -= t[10] * M[11] + t[2] * M[2] + t[7] * M[6];
        v[2] = t[10] * M[10] + t[4] * M[2] + t[9] * M[6];
        v[2] -= t[8] * M[6] + t[11] * M[10] + t[5] * M[2];
        v[3] = t[2] * M[10] + t[5] * M[11] + t[1] * M[6];
        v[3] -= t[4] * M[11] + t[0] * M[6] + t[3] * M[10];
        n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    }

    if (n < eps)
        cout << " n is " << n << "\n\n";
    return -1;

    n = sqrt(n);
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
    v[3] /= n;

    return 0;
}