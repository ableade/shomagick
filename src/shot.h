#ifndef SHOT_HPP_
#define SHOT_HPP_

#include "camera.h"

class Shot {
    private:
        std::string imageName;
        Camera camera;
        Pose cameraPose;

    public: 
        Shot():imageName(), camera (), cameraPose() {}
        Shot(std::string image, Camera camera, Pose pose) : imageName(image), camera(camera), cameraPose(pose) {}
        std::string getId() const {return this->imageName;}
        Camera getCamera() const {return this->camera;}
        const Pose& getPose() const {return this->cameraPose;}
        Pose getPose() { return this->cameraPose; }
        friend std::ostream & operator << (std::ostream &out, const Pose &p); 
};

#endif