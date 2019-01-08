#ifndef RECONSTRUCTION_HPP_
#define RECONSTRUCTION_HPP_

#include "shot.h"

class CloudPoint {

    private:
        int id;
        cv::Point3d position;
        cv::Scalar color;
        double projError;

    public:
        CloudPoint(): id(), position(), color(), projError() {}
        CloudPoint(int id, cv::Point3d position, cv::Scalar color, double projError): id(id), position(position) , color(color) , 
        projError(projError) {}
        int getId() const {return this->id;}
        double getError() const {return this->projError;}
        cv::Point3d getPosition() const {return this->position;}
        void setPosition(cv::Point3d pos) {this->position  = pos;}
        void setId(int id) {this->id = id;}
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
        void addCloudPoint(CloudPoint cPoint);
};

#endif