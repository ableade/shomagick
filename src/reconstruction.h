#pragma once
#include "shot.h"

const int BUNDLE_INTERVAL = 999999;
const double NEW_POINTS_RATIO = 1.2;

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
        void setColor(cv::Scalar col) {color = col;}
        cv::Scalar getColor() const { return color;}
        cv::Scalar getColor() { return color; }
};

class Reconstruction {
    private:
        std::map<std::string, Shot> shots;
        std::map<int, CloudPoint> cloudPoints;
        Camera camera;
        int lastPointCount;
        int lastShotCount;

    public:
        Reconstruction();
        Reconstruction(Camera camera);
        std::map<std::string, Shot>& getReconstructionShots();
        const std::map<std::string, Shot>& getReconstructionShots() const;
        bool hasShot(std::string shotId) const;
        const std::map<int, CloudPoint>& getCloudPoints() const;
        std::map<int, CloudPoint>& getCloudPoints();
        void addCloudPoint(CloudPoint cPoint);
        bool hasTrack(std::string trackId) const;
        const Camera& getCamera() const;
        void saveReconstruction(const std::string recFileName) const;
        Camera& getCamera();
        void updateLastCounts();
        bool needsBundling();
        bool needsRetriangulation();
};