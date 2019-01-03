#include "camera.h"
#include <ostream>
#include <algorithm>

using std::ostream;
using std::max;

cv::Mat Pose::getRotationMatrix() const
{
    cv::Mat rods;
    cv::Rodrigues(this->rotation, rods);
    return rods;
}

cv::Mat Pose::getOrigin() const
{
    cv::Mat tRot;
    auto origin = this->getRotationMatrix();
    cv::transpose(-origin , tRot);
    origin = tRot * this->translation;
    return origin;
}

ostream & operator << (ostream &out, const Pose &p) 
{ 
    out << "Rotation vector: " << p.rotation << std::endl;
    out << "Translation vector: " << p.translation <<std::endl; 
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
Camera::Camera() : cameraMatrix(), distortionCoefficients(), height(), width()
{
    int dimension = 3;
    this->cameraMatrix = cv::Mat::eye(dimension, dimension, CV_32F);
}

Camera::Camera(cv::Mat cameraMatrix, cv::Mat distortion, int height, int width) : cameraMatrix(cameraMatrix), distortionCoefficients(distortion), height(height),
width(width){}

cv::Mat Camera::getKMatrix() { return this->cameraMatrix; }

cv::Mat Camera::getDistortionMatrix()
{
    cv::Mat dist = (cv::Mat_<double>(5, 1) << 9.5451901612149271e-3, -5.4949250292936147e-3, 0., 0., 6.0371565989711740e-3);
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
	cv::convertPointsHomogeneous(points, hPoints);
	//std::vector<cv::Point2f> uPoints;
	cv::undistortPoints(points, points, this->cameraMatrix, this->getDistortionMatrix());
    opengv::bearingVector_t bearing;
    auto convPoint = hPoints[0];
    auto hPoint = cv::Vec3f(convPoint);
    l = std::sqrt(hPoint[0] * hPoint[0] + hPoint[1] * hPoint[1] + hPoint[2] * hPoint[2]);
    for (int j = 0; j < 3; ++j)
        bearing[j] = hPoint[j] / l;
    
    return bearing;
}

double Camera::getFocal() {
	return 0.6;
	//return this->cameraMatrix.at<double>(0, 0);
}

double Camera::getK1() {
	return -0.1;
	//return this->getDistortionMatrix().at<double>(0,0);
}

double Camera::getk2() {
	return -0.01;
	//return this->getDistortionMatrix().at<double>(1, 0);
}

cv::Point2f Camera::normalizeImageCoordinates(const cv::Point2f pixelCoords) const
{
	const auto size = max(width, height);

	const auto pixelX = pixelCoords.x;
	const auto pixelY = pixelCoords.y;
	const auto normX = ( (1.0f * width)  / size ) * (pixelX - width / 2.0f) / width;
	const auto normY = ( (1.0f * height) / size ) * (pixelY - height / 2.0f) / height;
	return {
		normX,
		normY,
	};
}

cv::Point2f Camera::denormalizeImageCoordinates(const cv::Point2f normalizedCoords) const
{
	const auto size = max(width, height);
	auto normX = normalizedCoords.x;
	auto normY = normalizedCoords.y;

	const auto pixelX = ( normX * width  * size / (1.0f * width) ) + width  / 2.0f;
	const auto pixelY = ( normY * height * size / (1.0f * height)) + height / 2.0f;

	return {pixelX, pixelY};
}

cv::Point2f Camera::projectBearing(opengv::bearingVector_t b) {
	auto x = b[0] / b[2];
	auto y = b[1] / b[2];

	auto r = x * x + y * y;
	auto radialDistortion = 1.0 + r * (getK1() + getk2() * r);
	
	return cv::Point2f{
		static_cast<float>( getFocal() * radialDistortion * x ),
		static_cast<float>(getFocal() * radialDistortion * y)
	};
}


