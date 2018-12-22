#include "camera.h"
#include <ostream>

using std::ostream;

cv::Mat Pose::getRotationMatrix() const
{
    cv::Mat rods;
    cv::Rodrigues(this->rotation, rods);
    return rods;
}

cv::Mat Pose::getOrigin() const
{
    cv::Mat tRot;
    cv::transpose(- this->rotation, tRot);
    cv::Mat origin = tRot * this->translation;
    return origin;
}

ostream & operator << (ostream &out, const Pose &p) 
{ 
    out << "Rotation " << p.rotation << std::endl;
    out << "Translation " << p.translation <<std::endl; 
    return out; 
} 

void Camera::_cvPointsToBearingVec(cv::Mat pRect, opengv::bearingVectors_t &bearings)
{
    double l;
    cv::Vec3f p;
    opengv::bearingVector_t bearing;
    for (auto i = 0; i < pRect.rows; ++i)
    {
        p = cv::Vec3f(pRect.row(i));
        l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        for (int j = 0; j < 3; ++j)
            bearing[j] = p[j] / l;
        bearings.push_back(bearing);
    }
}
Camera::Camera() : cameraMatrix(), distortionCoefficients()
{
    int dimension = 3;
    this->cameraMatrix = cv::Mat::eye(dimension, dimension, CV_32F);
}

Camera::Camera(cv::Mat cameraMatrix, cv::Mat distortion) : cameraMatrix(cameraMatrix), distortionCoefficients(distortion) {}

cv::Mat Camera::getKMatrix() { return this->cameraMatrix; }

cv::Mat Camera::getDistortionMatrix()
{
    cv::Mat dist = (cv::Mat_<double>(3, 3) << 9.5451901612149271e-3, -5.4949250292936147e-3, 0., 0., 6.0371565989711740e-3);
    //return this->distortionCoefficients;
    return dist;
}

void Camera::cvPointsToBearingVec(
    const std::vector<cv::Point2f> &points, opengv::bearingVectors_t &bearings)
{
    const int N1 = static_cast<int>(points.size());
    cv::Mat points1_mat = cv::Mat(points).reshape(1);

    // first rectify points and construct homogeneous points
    // construct homogeneous points
    cv::Mat ones_col1 = cv::Mat::ones(N1, 1, CV_32F);
    cv::hconcat(points1_mat, ones_col1, points1_mat);

    // undistort points
    cv::Mat points1_rect = points1_mat * this->cameraMatrix.inv();

    // compute bearings
    this->_cvPointsToBearingVec(points1_rect, bearings);
}

opengv::bearingVector_t Camera::cvPointToBearingVec(cv::Point2f &point)
{
    double l;
    std::vector<cv::Point2f> points {point};
    std::vector<cv::Point3f> hPoints;
    opengv::bearingVector_t bearing;
    cv::convertPointsHomogeneous(points, hPoints);
    auto convPoint = hPoints[0];
    auto hPoint = cv::Vec3f(convPoint);
    l = std::sqrt(hPoint[0] * hPoint[0] + hPoint[1] * hPoint[1] + hPoint[2] * hPoint[2]);
    for (int j = 0; j < 3; ++j)
        bearing[j] = hPoint[j] / l;
    
    return bearing;
}
