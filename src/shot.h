#ifndef SHOT_HPP_
#define SHOT_HPP_

#include "camera.h"

class Shot {
    private:
        std::string image;
        Camera camera;
        Pose cameraPose;

    public: 
        Shot():image(), camera (), cameraPose() {}
        Shot(std::string image, Camera camera, Pose pose) : image(image), camera(camera), cameraPose(pose) {}
        std::string getId() const {return this->image;}
        Camera getCamera() const {return this->camera;}
        const Pose& getPose() const {return this->cameraPose;} 
};

#endif