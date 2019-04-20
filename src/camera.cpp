#include "camera.h"
#include <ostream>
#include <algorithm>
#include <string>
#include <Eigen/Core>
//#include <boost/filesystem.hpp>

using std::ostream;
using std::max;
using std::vector;
using std::string;
using cv::Matx33d;
using cv::Mat;
using cv::Vec3d;
using cv::Point2d;
using cv::Size;
using opengv::bearingVector_t;
using opengv::bearingVectors_t;

Matx33d Pose::getRotationMatrix() const
{
    Mat rods;
    Rodrigues(this->rotation, rods);
    return rods;
}

ShoColumnVector3d Pose::getRotationVector() const
{
    return rotation;
}

cv::Mat Pose::getRotationMatrixInverse() const {
    auto r = getRotationMatrix();
    Mat tR;
    transpose(r, tR);
    return tR;
}

ShoColumnVector3d Pose::getOrigin() const
{
    Mat tRot;
    auto origin = -(this->getRotationMatrix());
   // std::cout << "Rotation matrix is " << origin << std::endl;

    //std::cout << "Rotation matrix transpose is " << tRot << std::endl;
    auto t = Mat(origin.t() * Mat(this->translation));
    return t;
}

ShoColumnVector3d Pose::getTranslation() const{
    return translation;
}

void Pose::setTranslation(ShoColumnVector3d t)
{
    translation = t;
}

void Pose::setRotationVector(cv::Mat src) {
    ShoColumnVector3d rotationVector;
    if (src.rows == 3 && src.cols == 3) {
        //src is currently a rotation matrix
        cv::Rodrigues(src, src);
    }
    rotation = src;
}

Pose Pose::poseInverse() const {
    auto inv = Pose{};
    auto r = getRotationMatrix();
    std::cout << "R was " << r << "\n\n";
    cv::Matx33d transposedRotation;
    cv::transpose(r, transposedRotation);
    //cv::Rodrigues()
    inv.setRotationVector(Mat(transposedRotation));
    cv::transpose(-r, transposedRotation);
    auto inverseTranslation = transposedRotation * Mat(translation);
    inv.setTranslation(inverseTranslation);
    return inv;
}

Pose Pose::compose(const Pose& other) const {
    auto r = getRotationMatrix();
    auto otherR = other.getRotationMatrix();
    auto nR = (r * otherR);
    auto nT = (r * Mat(other.getTranslation())) + getTranslation();
    Pose p;
    p.setTranslation(nT);
    p.setRotationVector(Mat(nR));
    return p;
}

ostream & operator << (ostream &out, const Pose &p)
{
    out << "Rotation vector: " << p.rotation << std::endl;
    out << "Translation vector: " << p.translation << std::endl;
    return out;
}

void Camera::_cvPointsToBearingVec(cv::Mat pRect, opengv::bearingVectors_t &bearings) const
{
    double l;
    Vec3d p;
    opengv::bearingVector_t bearing;
    for (auto i = 0; i < pRect.rows; ++i)
    {
        p = cv::Vec3d(pRect.row(i));
        l = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        for (int j = 0; j < 3; ++j)
            bearing[j] = p[j] / l;
        bearings.push_back(bearing);
    }
}
Camera::Camera() : cameraMatrix(), distortionCoefficients(), height(), width(),  cameraMake(),
    cameraModel(), initialK1(), initialK2(), initialPhysicalFocal()
{
    int dimension = 3;
    this->cameraMatrix = cv::Mat::eye(dimension, dimension, CV_32F);
    this->cameraMake = "";
    this->cameraModel = "";
    this->initialPhysicalFocal = 0.0;
    this->initialK1 = 0.0;
    this->initialK2 = 0.0;
}

Camera::Camera(Mat cameraMatrix, Mat distortion, int height, int width, int scaledHeight, int scaledWidth) : 
    scaledHeight(scaledHeight), scaledWidth(scaledWidth), cameraMatrix(cameraMatrix), distortionCoefficients(distortion), height(height),
width(width),cameraMake(),
cameraModel(),  initialK1(),  initialK2(), initialPhysicalFocal() {
    assert (!cameraMatrix.empty());
    assert(!distortionCoefficients.empty());
    assert(height !=0 && width != 0);
    this->initialK1 = this->getK1();
    this->initialK2 = this->getK2();
    this->initialPhysicalFocal = this->getPhysicalFocalLength();
}

Mat Camera::getKMatrix() { return this->cameraMatrix; }

Mat Camera::getNormalizedKMatrix() const {
    auto lensSize = this->getPhysicalFocalLength();

    Mat normK = (cv::Mat_<double>(3, 3) <<
        lensSize,   0.,          0.,
        0.,         lensSize,    0,
        0.,          0.,         1);

    std::cout << "Normalized k matrix is " << normK << "\n";
    return normK;
}

Mat Camera::getDistortionMatrix() const
{
    if( distortionCoefficients.rows ==1)
        return distortionCoefficients.t();

    return distortionCoefficients;
}

void Camera::cvPointsToBearingVec(const vector<Point2d> &points, bearingVectors_t &bearings) const
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
    _cvPointsToBearingVec(points1_rect, bearings);
}

opengv::bearingVectors_t Camera::normalizedPointsToBearingVec(const std::vector<cv::Point2d>& points) const
{
    opengv::bearingVectors_t bearings;
    for (const auto point : points) {
        auto bearing = normalizedPointToBearingVec(point);
        bearings.push_back(bearing);
    }
    return bearings;
}

bearingVector_t Camera::normalizedPointToBearingVec(const Point2d &point) const
{
    std::vector<cv::Point2d> points{ point };
    std::vector<cv::Point3d> hPoints;
    cv::undistortPoints(points, points, this->getNormalizedKMatrix(), this->getDistortionMatrix());
    cv::convertPointsHomogeneous(points, hPoints);
    opengv::bearingVector_t bearing;
    auto convPoint = hPoints[0];
    auto hPoint = cv::Vec3d(convPoint);
    const double l = std::sqrt(hPoint[0] * hPoint[0] + hPoint[1] * hPoint[1] + hPoint[2] * hPoint[2]);    
    for (int j = 0; j < 3; ++j)
        bearing[j] = hPoint[j] / l;

    return bearing;
}

double Camera::getPixelFocal() const{
    return this->cameraMatrix.at<double>(0, 0);
}

void Camera::setPixelFocal(double pixelFocal)
{
    CV_Assert(pixelFocal > 1);
    cameraMatrix.at<double>(0, 0) = pixelFocal;
}

double & Camera::getK1()
{
    return this->getDistortionMatrix().at<double>(0, 0);
}

double & Camera::getK2() {
    return this->getDistortionMatrix().at<double>(1, 0);
}

double Camera::getPhysicalFocalLength() const {
    auto h = (scaledHeight) ? scaledHeight : height;
    auto w = (scaledWidth) ? scaledWidth : width;

    return (double)this->getPixelFocal() / (double)max(h, w);
}

double Camera::getK1() const {
    return this->getDistortionMatrix().at<double>(0,0);
}

double Camera::getK2() const{
    return this->getDistortionMatrix().at<double>(1, 0);
}

double Camera::getInitialK1() const {
    return this->initialK1;
}

double Camera::getInitialK2() const {
    return this->initialK2;
}

double Camera::getInitialPhysicalFocal() const {
    return this->initialPhysicalFocal;
}

Point2d Camera::normalizeImageCoordinate(const cv::Point2d pixelCoords) const
{
    auto h = (scaledHeight) ? scaledHeight : height;
    auto w = (scaledWidth) ? scaledWidth : width;

    const auto size = max(w, h);

    float step = 0.5;
    const auto pixelX = pixelCoords.x + step;
    const auto pixelY = pixelCoords.y + step;
    const auto normX = ((1.0f * w) / size) * (pixelX - w / 2.0f) / w;
    const auto normY = ((1.0f * h) / size) * (pixelY - h / 2.0f) / h;

    return {
        normX,
        normY,
    };
}

vector<Point2d> Camera::normalizeImageCoordinates(const vector<Point2d>& points) const
{
    std::vector<cv::Point2d> results;

    for (const auto& point : points) {
        results.push_back(normalizeImageCoordinate(point));
    }
    
    return results;
}

Point2d Camera::denormalizeImageCoordinates(const Point2d normalizedCoords) const
{
    auto h = (scaledHeight) ? scaledHeight : height;
    auto w = (scaledWidth) ? scaledWidth : width;

    const auto size = max(w, h);
    auto normX = normalizedCoords.x;
    auto normY = normalizedCoords.y;

    float pixelX = ((normX * width  * size / (1.0f * w)) + w / 2.0f) - 0.5;
    float pixelY = ((normY * height * size / (1.0f * h)) + h / 2.0f) -0.5;

    return { pixelX, pixelY };
}

 vector<Point2d> Camera::denormalizeImageCoordinates(const vector<Point2d>& points) const {
     std::vector<cv::Point2d> results;
     for(const auto point: points) {
         results.push_back(denormalizeImageCoordinates(point));
     }
     return results;
 }

Point2d Camera::projectBearing(bearingVector_t b) {
    auto x = b[0] / b[2];
    auto y = b[1] / b[2];

    auto r = x * x + y * y;
    auto radialDistortion = 1.0 + r * (getK1() + getK2() * r);

    return cv::Point2d{
        static_cast<float>(getPhysicalFocalLength() * radialDistortion * x),
        static_cast<float>(getPhysicalFocalLength() * radialDistortion * y)
    };
}

Camera Camera::getCameraFromCalibrationFile(string calibrationFile) {
    int height, width;
    cv::Mat cameraMatrix, distortionParameters;
    //assert(boost::filesystem::exists(calibrationFile));
    cv::FileStorage fs(calibrationFile, cv::FileStorage::READ);
    fs["image_height"] >> height;
    fs["image_width"] >> width;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distortionParameters;
    return Camera{ cameraMatrix, distortionParameters, height, width };
}

Camera Camera::getCameraFromExifMetaData(std::string image)
{
    return Camera();
}

void Camera::setFocalWithPhysical(double physicalFocal)
{
   auto pixelFocal = physicalFocal * (double)max(this->height, this->width);
   setPixelFocal(pixelFocal);
}

void Camera::setK1(double k1)
{
    getK1() = k1;
}

void Camera::setK2(double k2)
{
    getK2() = k2;
}

void Camera::setScaledHeight(int h)
{
    scaledHeight = h;
}

void Camera::setScaledWidth(int w)
{
    scaledWidth = w;
}

int Camera::getHeight()
{
    return height;
}

int Camera::getScaledHeight()
{
    return scaledHeight;
}

int Camera::getScaledWidth()
{
    return scaledWidth;
}

int Camera::getWidth()
{
    return width;
}


