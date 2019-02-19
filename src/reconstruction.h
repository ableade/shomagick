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
        void setError(double projError) { this->projError = projError; }
        const cv::Point3d& getPosition() const {return this->position;}
        void setPosition(cv::Point3d pos) {this->position  = pos;}
        void setId(int id) {this->id = id;}
        cv::Point3d& getPosition() { return this->position; }
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
        const std::map<std::string, Shot>& getReconstructionShots() const;
        bool hasShot(std::string shotId) const;
        const std::map<int, CloudPoint>& getCloudPoints() const;
        std::map<int, CloudPoint>& getCloudPoints();
        void addCloudPoint(CloudPoint cPoint);
        bool hasTrack(std::string trackId) const;
        const Camera& getCamera() const;
        Camera& getCamera();
};

#endif