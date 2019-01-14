#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <opencv2/opencv.hpp>
#include <opengv/types.hpp>
#include <ostream>

class Pose {
  private:
    cv::Mat rotation;
    cv::Mat translation;

  public:
    Pose () : rotation(cv::Mat::zeros(3,1,CV_32F)), translation(cv::Mat::zeros(3,1,CV_32F)) {}
    Pose(cv::Mat rotation, cv::Mat translation) : rotation(rotation), translation(translation) { setRotationVector(rotation); }
    cv::Mat getRotationMatrix() const;
    cv::Mat getRotationVector() const;
    void setRotationVector(cv::Mat rot);
    void setTranslation(cv::Mat rot);
    cv::Mat getOrigin() const;
    friend std::ostream & operator << (std::ostream& out, const Pose& p);
    Pose inverse() const;
    Pose compose(const Pose& p) const;
    cv::Mat getTranslation() const;
};

class Camera
{

  private:
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
    void _cvPointsToBearingVec(cv::Mat pRect, opengv::bearingVectors_t& );
    int height;
    int width;
    std::string cameraMake;
    std::string cameraModel;
    float initialK1;
    float initialK2;
    float initialPhysicalFocal;

  public:
    Camera();
    Camera(cv::Mat cameraMatrix, cv::Mat distortion, int height =0, int width =0);
    double getPixelFocal() const;
    double getPhysicalFocalLength() const;
    cv::Mat getKMatrix();
    cv::Mat getNormalizedKMatrix() const;
    cv::Mat getDistortionMatrix() const;
    void cvPointsToBearingVec(
    const std::vector<cv::Point2f>&, opengv::bearingVectors_t& );
    opengv::bearingVector_t  normalizedPointToBearingVec(const cv::Point2f &point) const;
    cv::Point2f projectBearing(opengv::bearingVector_t);
    double getK1() const;
    double getK2() const;
    double getInitialK1() const;
    double getInitialK2() const;
    double getInitialPhysicalFocal() const;
    cv::Point2f normalizeImageCoordinates(const cv::Point2f) const;
    cv::Point2f denormalizeImageCoordinates(const cv::Point2f) const;
    static Camera getCameraFromCalibrationFile(std::string calibFile);
    static Camera getCameraFromExifMetaData(std::string image);
};
#endif