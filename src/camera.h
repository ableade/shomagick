#ifndef CAMERA_HPP_
#define CAMERA_HPP_

#include <opencv2/opencv.hpp>

class Camera
{

  private:
    cv::Mat cameraMatrix;

  public:
    Camera();
    Camera(cv::Mat cameraMatrix);
    double getFocal();
    cv::Mat getCameraMatrix();
};
#endif