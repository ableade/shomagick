#include "utilities.h"
#include <iostream>

using std::cerr;
using cv::Point3d;
using cv::Vec3d;
using std::string;

ShoRowVector3d convertPointToRowVector(Point3d p) {
    return { p.x, p.y, p.z };
}

ShoRowVector3d convertVecToRowVector(Vec3d p) {
    return { p(0), p(1), p(2) };
}

ShoColumnVector3d convertVecToColumnVector(Vec3d p) {
    return { p(0), p(1), p(2) };
}
ShoRowVector4d convertColumnVecToRowVector(ShoColumnVector3d a) {
    return { a(0,0), a(1,0), a(2,0) };
}


bool checkIfCudaEnabled()
{
    // ORB is the default feature detector
    auto cudaEnabled = cv::cuda::getCudaEnabledDeviceCount();
    if (cudaEnabled != -1 && cudaEnabled != 0) {
        cerr << "CUDA device detected. Running CUDA \n";
        cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
        return true;
    }
    else if (cudaEnabled == 0) {
        cerr << "Please recompile Open CV with CUDA support \n";
    }
    else if (cudaEnabled == -1) {
        cerr << "The CUDA driver is not installed, or is incompatible \n";
    }
    return false;
}

string getDateTimeAsString()
{
#if 0
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
#endif

#if 1
    unsigned long int sec = time(NULL);
    return std::to_string(sec);
#endif
}

