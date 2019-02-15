#include "reconstruction.h"
#include "reconstructor.h"
#include "map"
#include "string"

using std::map;
using std::string;

Reconstruction::Reconstruction() : shots(), cloudPoints(), camera() {}

Reconstruction::Reconstruction(std::map<std::string, Shot> shots, std::map<int, CloudPoint> cloudPoints, Camera camera) : shots(shots), cloudPoints(cloudPoints), camera(camera) {}

map<string, Shot>& Reconstruction::getReconstructionShots()
{
    return shots;
}

const map<string, Shot>& Reconstruction::getReconstructionShots() const
{
    return shots;
}

const map<int, CloudPoint>& Reconstruction::getCloudPoints() const
{
    return cloudPoints;
}

std::map<int, CloudPoint>& Reconstruction::getCloudPoints()
{
    return cloudPoints;
}

bool Reconstruction::hasShot(string shotId) const
{
    return this->shots.find(shotId) != this->shots.end();
}

void Reconstruction::addCloudPoint(CloudPoint cp) {
    this->cloudPoints[cp.getId()] = cp;
}

bool Reconstruction::hasTrack(std::string trackId) const
{
    return cloudPoints.find(stoi(trackId)) != cloudPoints.end();
}

const Camera& Reconstruction::getCamera() const
{
    return camera;
}

Camera& Reconstruction::getCamera() {
    return camera;
}

