#include "reconstruction.h"
#include "map"
#include "string"

using std::map;
using std::string;

Reconstruction::Reconstruction() : shots(), cloudPoints(), camera() {}

Reconstruction::Reconstruction(std::map<std::string, Shot> shots, std::map<int, CloudPoint> cloudPoints, Camera camera) : shots(shots), cloudPoints(cloudPoints), camera(camera) {}

map<string, Shot> &Reconstruction::getReconstructionShots()
{
    return this->shots;
}

map<int, CloudPoint> &Reconstruction::getCloudPoints()
{
    return this->cloudPoints;
}

bool Reconstruction::hasShot(string shotId)
{
    return this->shots.find(shotId) != this->shots.end();
}

void Reconstruction::addCloudPoint(CloudPoint cp) {
    this->cloudPoints[cp.getId()] = cp;
}