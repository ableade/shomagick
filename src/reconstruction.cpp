#include "reconstruction.h"
#include "reconstructor.h"
#include <fstream>
#include <map>
#include <string>

using std::map;
using std::string;
using std::ofstream;
using std::ios;

Reconstruction::Reconstruction() : shots(), cloudPoints(), camera(), lastPointCount(), lastShotCount() {}

Reconstruction::Reconstruction(Camera camera) :camera(camera), lastPointCount(), lastShotCount() {}

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
    cloudPoints[cp.getId()] = cp;
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

void Reconstruction::updateLastCounts()
{
    lastPointCount = cloudPoints.size();
    lastShotCount = shots.size();
}

bool Reconstruction::needsBundling()
{
    if (lastPointCount == 0 || lastShotCount == 0)
        return false;

    auto maxPoints = lastPointCount * NEW_POINTS_RATIO;
    auto maxShots = lastShotCount + BUNDLE_INTERVAL;
    
    return (cloudPoints.size() >= maxPoints || shots.size() >= maxShots);
}

bool Reconstruction::needsRetriangulation()
{
    if (lastPointCount == 0 || lastShotCount == 0)
        return false;

    auto maxPoints = lastPointCount * NEW_POINTS_RATIO;
    return (cloudPoints.size() > maxPoints);
}

void Reconstruction::saveReconstruction(const string recFileName) const
{
    ofstream recFile(recFileName);
    recFile << "ply\n";
    recFile << "format ascii 1.0\n";
    recFile << "element vertex "<< cloudPoints.size()<< "\n";
    recFile << "property float x\n";
    recFile << "property float y\n";
    recFile << "property float z\n";
    recFile << "property uchar diffuse_red\n";
    recFile << "property uchar diffuse_green\n";
    recFile << "property uchar diffuse_blue\n";
    recFile << "end_header\n";
    for (const auto [trackId, cp] : cloudPoints) {
        recFile << cp.getPosition().x << " " << cp.getPosition().y << " " << cp.getPosition().z << " " << 
            cp.getColor()[0] << " " <<cp.getColor()[1]<< " " 
            << cp.getColor()[2] << "\n";
    }
    recFile.close();
}

