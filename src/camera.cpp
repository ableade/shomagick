#include "camera.h"
#include <ostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include "json.hpp"
#include "allclose.h"
//#include <boost/filesystem.hpp>

using std::ostream;
using std::max;
using std::vector;
using std::string;
using std::ifstream;
using cv::Matx33d;
using cv::Mat;
using cv::Vec3d;
using cv::Point2d;
using cv::Point_;
using cv::Size;
using opengv::bearingVector_t;
using opengv::bearingVectors_t;
using json = nlohmann::json;

#ifdef _MSC_VER 

#endif

Matx33d Pose::getRotationMatrix() const
{
    Mat rods;
    Rodrigues(rotation, rods);
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
    auto origin = -(getRotationMatrix());
    auto t = Mat(origin.t() * Mat(translation));
    return t;
}

ShoColumnVector3d Pose::getTranslation() const {
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
       // CV_Assert(allClose(cv::determinant(src), 1));
        //CV_Assert(allClose(src.inv(), src.t()));
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

std::ostream & operator<<(std::ostream & out, const Camera & c)
{
    out << "Initial physical focal " << c.getInitialPhysicalFocal() << "\n";
    out << "Physical focal " << c.getPhysicalFocalLength() << "\n";
    out << "Initial K1 " << c.getInitialK1() << "\n";
    out << "Initial K2 " << c.getInitialK2() << "\n";
    out << "K1 " << c.getK1() << "\n";
    out << "K2 " << c.getK2() << "\n";
    out << "Height " << c.getHeight() << "\n";
    out << "Width " << c.getWidth() << "\n";
    out << "Scaled height " << c.getScaledHeight() << "\n";
    out << "Scaled width " << c.getScaledWidth() << "\n";
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
Camera::Camera() : cameraMatrix(), distortionCoefficients(), height_(), width_(), cameraMake_(),
cameraModel_(), initialK1_(), initialK2_(), initialPhysicalFocal()
{
    int dimension = 3;
    this->cameraMatrix = cv::Mat::eye(dimension, dimension, CV_32F);
    this->cameraMake_ = "";
    this->cameraModel_ = "";
    this->initialPhysicalFocal = 0.0;
    this->initialK1_ = 0.0;
    this->initialK2_ = 0.0;
}

Camera::Camera(Mat cameraMatrix, Mat distortion, int height, int width, int scaledHeight, int scaledWidth) :
    scaledHeight_(scaledHeight), scaledWidth_(scaledWidth), cameraMatrix(cameraMatrix), distortionCoefficients(distortion), height_(height),
    width_(width), cameraMake_(), 
    cameraModel_(), initialK1_(), initialK2_(), initialPhysicalFocal() {
    assert(!cameraMatrix.empty());
    assert(!distortionCoefficients.empty());
    assert(height != 0 && width != 0);
    initialK1_ = this->getK1();
    initialK2_ = this->getK2();
    initialPhysicalFocal = getPhysicalFocalLength();
}

Mat Camera::getKMatrix() { return this->cameraMatrix; }

Mat Camera::getNormalizedKMatrix() const {
    auto lensSize = getPhysicalFocalLength();

    Mat normK = (cv::Mat_<double>(3, 3) <<
        lensSize, 0., 0.,
        0., lensSize, 0,
        0., 0., 1);

    return normK;
}

Mat Camera::getDistortionMatrix() const
{
    if (distortionCoefficients.rows == 1)
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

double Camera::getPixelFocal() const {
    return this->cameraMatrix.at<double>(0, 0);
}

void Camera::setPixelFocal(double pixelFocal)
{
    CV_Assert(pixelFocal > 1);
    cameraMatrix.at<double>(0, 0) = pixelFocal;
    cameraMatrix.at<double>(1, 1) = pixelFocal;
}

double & Camera::getK1()
{
    return getDistortionMatrix().at<double>(0, 0);
}

double & Camera::getK2() {
    return this->getDistortionMatrix().at<double>(1, 0);
}

double Camera::getPhysicalFocalLength() const {

    return (double)getPixelFocal() / (double)max(getHeight(), getWidth());
}

double Camera::getK1() const {
    return this->getDistortionMatrix().at<double>(0, 0);
}

double Camera::getK2() const {
    return this->getDistortionMatrix().at<double>(1, 0);
}

double Camera::getInitialK1() const {
    return this->initialK1_;
}

double Camera::getInitialK2() const {
    return this->initialK2_;
}

double Camera::getInitialPhysicalFocal() const {
    return this->initialPhysicalFocal;
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

Camera Camera::getCameraFromExifMetaData(ImageMetadata metadata)
{
    auto focal = metadata.focalRatio;
    if (focal == 0)
        focal = 0.75;
    
    focal *= max(metadata.width, metadata.height); //Use this as default focal for camera

    cv::Mat dist = (cv::Mat_<double>(4, 1) <<
        0,
        0,
        0.,
        0.
        );

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        focal,
        0.,
        metadata.width / 2,
        0.,
        focal,
        metadata.height / 2,
        0.,
        0.,
        1
        );

    return Camera(cameraMatrix, dist, metadata.height, metadata.width);
}

void Camera::setFocalWithPhysical(double physicalFocal)
{
    auto pixelFocal = physicalFocal * (double)max(height_, width_);
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
    scaledHeight_ = h;
}

void Camera::setScaledWidth(int w)
{
    scaledWidth_ = w;
}

int Camera::getHeight() const
{
    return height_;
}

int Camera::getScaledHeight() const
{
    return scaledHeight_;
}

int Camera::getScaledWidth() const
{
    return scaledWidth_;
}

int Camera::getWidth() const
{
    return width_;
}