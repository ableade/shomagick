#ifndef RECONSTRUCTION_HPP_
#define RECONSTRUCTION_HPP_

#include "shot.h"

class CloudPoint {

    private:
        cv::Point3d position;
        cv::Scalar color;
        int id;
        double projError;

    public:
        CloudPoint():position(), color(), id(), projError() {}
        CloudPoint(cv::Point3d position, cv::Scalar color, int id, double projError): position(position) , color(color) , id(id),
        projError(projError) {}
        int getId() const {return this->id;}
        double getError() const {return this->projError;}
        cv::Point3d getPosition() const {return this->position;}
};

class Reconstruction {
    private:
        std::map<std::string, Shot> shots;
        std::map<int, CloudPoint> cloudPoints;
        Camera camera;

    public:
        Reconstruction();
        Reconstruction(std::map<std::string, Shot> shots, std::map<int, CloudPoint> cloudPoints, Camera camera);
        std::map<std::string, Shot>& getReconstructionShots();
        bool hasShot(std::string shotId);
        std::map<int, CloudPoint>& getCloudPoints();
};

#endif