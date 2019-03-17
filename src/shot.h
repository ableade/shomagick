#ifndef SHOT_HPP_
#define SHOT_HPP_

#include "camera.h"
#include "image.hpp"
#include "flightsession.h"

struct ShotMetadata {
    cv::Point3d gpsPosition;
    double gpsDop;
    double captureTime;
    int orientation;
    ShotMetadata();
    ShotMetadata(cv::Point3d gpsPosition, double dop, double captureTime, int orientation);
    ShotMetadata(ImageMetadata imageExif, const FlightSession& flight);
};

class Shot {
    private:
        std::string imageName;
        Camera camera;
        Pose cameraPose;
        ShotMetadata metadata;

    public: 
        Shot():imageName(), camera (), cameraPose(), metadata(){}
        Shot(std::string imaeName, Camera camera, Pose pose) : imageName(imageName), camera(camera),
            cameraPose(pose) {}
        Shot(std::string image, Camera camera, Pose pose, ShotMetadata metadata) : imageName(image)
            , camera(camera), cameraPose(pose), metadata(metadata) {}
        std::string getId() const {return this->imageName;}
        Camera getCamera() const {return this->camera;}
        const Pose& getPose() const {return this->cameraPose;}
        Pose getPose() { return this->cameraPose; }
        const ShotMetadata getMetadata() const { return metadata; }
        std::tuple<ShoRowVector3d, ShoRowVector3d, ShoRowVector3d> getOrientationVectors() const;
        friend std::ostream & operator << (std::ostream &out, const Pose &p); 
};

#endif