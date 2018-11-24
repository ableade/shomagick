#include "camera.h"

Camera::Camera()
{
    int dimension = 3;
    this->cameraMatrix = cv::Mat::eye(dimension, dimension, CV_32F);
}

Camera::Camera(cv::Mat cameraMatrix) : cameraMatrix(cameraMatrix){};