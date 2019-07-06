#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <opencv2/opencv.hpp>
#include <opengv/types.hpp>
#include <ostream>
#include "types.h"

class Pose {
  private:
    ShoColumnVector3d translation_;
    ShoColumnVector3d rotation_; //Rotation is stored in short format as a vec 3d (3 channel Mat)

  public:
    Pose() : rotation_({ 0,0,0 }), translation_({ 0,0,0 }) {}
    Pose(cv::Mat rotation, cv::Vec3d translation) : translation_(translation) { setRotationVector(rotation); }
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
    int height_;
    int width_;
    int scaledHeight_;
    int scaledWidth_;
    std::string cameraMake_;
    std::string cameraModel_;
    float initialK1_;
    float initialK2_;
    float initialPhysicalFocal;

  public:
    Camera();
    Camera(cv::Mat cameraMatrix, cv::Mat distortion, int height =0, int width =0, int scaledHeight =0, int scaledWidth =0);
    double getPixelFocal() const;

    double getPhysicalFocalLength() const;
    cv::Mat getKMatrix();
    cv::Mat getNormalizedKMatrix() const;
    cv::Mat getDistortionMatrix() const;
    void cvPointsToBearingVec(
    const std::vector<cv::Point2d>&, opengv::bearingVectors_t& ) const;
    template <typename T>
    opengv::bearingVectors_t normalizedPointsToBearingVec(const std::vector<cv::Point_<T>>& points) const;
    template <typename T>
    opengv::bearingVector_t  normalizedPointToBearingVec(const cv::Point_<T>& point) const;
    cv::Point2d projectBearing(opengv::bearingVector_t);
    double getK1() const;
    double getK2() const;
    double getInitialK1() const;
    double getInitialK2() const;
    double getInitialPhysicalFocal() const;
    template <typename T>
    cv::Point_<T> normalizeImageCoordinate(const cv::Point_<T>&) const;
    template <typename T>
    cv::Point_<T>denormalizeImageCoordinates(const cv::Point_<T>&) const;
    template<typename T>
    std::vector<cv::Point_<T>> normalizeImageCoordinates(const std::vector<cv::Point_<T>>&) const;
    template <typename T>
    std::vector<cv::Point_<T>> denormalizeImageCoordinates(const std::vector<cv::Point_<T>>&) const;
    friend std::ostream & operator << (std::ostream& out, const  Camera & c);
    static Camera getCameraFromCalibrationFile(std::string calibFile);
    static Camera getCameraFromExifMetaData(std::string image);
    void setFocalWithPhysical(double physicalFocal);
    void setK1(double k1);
    void setK2(double k2);
    void setScaledHeight(int h);
    void setScaledWidth(int w);
    int getHeight() const;
    int getScaledHeight() const;
    int getScaledWidth() const;
    int getWidth() const;
    void setPixelFocal(double pixelFocal);
    double& getK1();
    double&  getK2();
};

#include "camera.inl"
#endif