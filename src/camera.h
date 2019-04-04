#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <opencv2/opencv.hpp>
#include <opengv/types.hpp>
#include <ostream>
#include "types.h"


class Pose {
  private:
    ShoColumnVector3d translation;
    ShoColumnVector3d rotation; //Rotation is stored in short format as a vec 3d (3 channel Mat)

  public:
    Pose() : rotation({ 0,0,0 }), translation({ 0,0,0 }) {}
    Pose(cv::Mat rotation, cv::Vec3d translation) : translation(translation) { setRotationVector(rotation); }
    cv::Matx33d getRotationMatrix() const;
    ShoColumnVector3d getRotationVector() const;
    cv::Mat getRotationMatrixInverse() const;
    void setRotationVector(cv::Mat rot);
    void setTranslation(ShoColumnVector3d t);
    ShoColumnVector3d getOrigin() const;
    friend std::ostream & operator << (std::ostream& out, const Pose& p);
    Pose poseInverse() const;
    Pose compose(const Pose& p) const;
    ShoColumnVector3d getTranslation() const;
};

class Camera
{

  private:
    cv::Mat cameraMatrix;
    cv::Mat_<double> distortionCoefficients;
    void _cvPointsToBearingVec(cv::Mat pRect, opengv::bearingVectors_t& ) const;
    int height;
    int width;
    int scaledHeight;
    int scaledWidth;
    std::string cameraMake;
    std::string cameraModel;
    float initialK1;
    float initialK2;
    float initialPhysicalFocal;

  public:
    Camera();
    Camera(cv::Mat cameraMatrix, cv::Mat distortion, int height =0, int width =0, int scaledHeight =0, int scaledWidth =0);
    const double& getPixelFocal() const;

    double getPhysicalFocalLength() const;
    cv::Mat getKMatrix();
    cv::Mat getNormalizedKMatrix() const;
    cv::Mat getDistortionMatrix() const;
    void cvPointsToBearingVec(
    const std::vector<cv::Point2f>&, opengv::bearingVectors_t& ) const;
    opengv::bearingVector_t  normalizedPointToBearingVec(const cv::Point2f &point) const;
    cv::Point2f projectBearing(opengv::bearingVector_t);
    const double& getK1() const;
    const double& getK2() const;
    double getInitialK1() const;
    double getInitialK2() const;
    double getInitialPhysicalFocal() const;
    cv::Point2f normalizeImageCoordinate(const cv::Point2f) const;
    std::vector<cv::Point2f> normalizeImageCoordinates(const std::vector<cv::Point2f>&) const;
    cv::Point2f denormalizeImageCoordinates(const cv::Point2f) const;
    std::vector<cv::Point2f> denormalizeImageCoordinates(const std::vector<cv::Point2f>&) const;
    static Camera getCameraFromCalibrationFile(std::string calibFile);
    static Camera getCameraFromExifMetaData(std::string image);
    void setFocalWithPhysical(double physicalFocal);
    void setK1(double k1);
    void setK2(double k2);
    void setScaledHeight(int h);
    void setScaledWidth(int w);
    int getHeight();
    int getScaledHeight();
    int getScaledWidth();
    int getWidth();
    double& getPixelFocal();
    double& getK1();
    double&  getK2();
};
#endif
