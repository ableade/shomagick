#pragma once
#include "shot.h"
#include "flightsession.h"

const int BUNDLE_INTERVAL = 999999;
const double NEW_POINTS_RATIO = 1.2;

class CloudPoint {

    private:
        int id;
        cv::Point3d position;
        cv::Scalar color;
        std::map<std::string, Eigen::VectorXd> projErrors_;

    public:
        CloudPoint(): id(), position(), color(), projErrors_() {}
        CloudPoint(int id, cv::Point3d position, cv::Scalar color, std::map<std::string, Eigen::VectorXd> projErrors): id(id), position(position) , color(color) ,
        projErrors_(projErrors) {}
        int getId() const {return this->id;}
        std::map<std::string, Eigen::VectorXd> getError() const {return projErrors_;}
        void setError(std::map<std::string, Eigen::VectorXd>  projError) { projErrors_ = projError; }
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
        bool usesGPS = false;

    public:
        Reconstruction();
        Reconstruction(Camera camera);
        void addShot(std::string shotId, Shot shot);
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
        void mergeReconstruction(const Reconstruction& rec);
        void alignToGps();
        void applySimilarity(double s, cv::Matx33d a, ShoColumnVector3d b);
        void setGPS(bool useGps);
        const Shot& getShot(std::string shotId) const;
        Shot& getShot(std::string shotId);
        bool usesGps() const;
        std::tuple<double, cv::Matx33d, ShoColumnVector3d> getGPSTransform();
};