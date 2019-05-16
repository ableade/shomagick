#include "reconstruction.h"
#include "reconstructor.h"
#include "utilities.h"
#include <fstream>
#include <map>
#include <string>

using std::map;
using std::string;
using std::ofstream;
using std::ios;
using std::set;
using cv::Mat;
using cv::Matx33d;
using cv::Point3d;

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

void Reconstruction::mergeReconstruction(const Reconstruction & rec)
{
    set<int> commonTracks;
    for (const auto &[id, cp] : rec.getCloudPoints()) {
        if (getCloudPoints().find(id) != getCloudPoints().end()) {
            //This is a common track with the current reconstruction
            commonTracks.insert(id);
        }
    }

    Mat p1, p2;
    if (commonTracks.size() > 6) {
        std::cout << "Size of common tracks is " << commonTracks.size() << "\n";
        for (auto track : commonTracks) {
            p1.push_back(cloudPoints.at(track).getPosition());
            p2.push_back(rec.getCloudPoints().at(track).getPosition());
        }
        std::cout << "P1 is " << p1 << "\n";
        std::cout << "P2 is " << p2 << "\n";
    }
 
    Mat t;
    Mat inliers;

    cv::estimateAffine3D(p1, p2,t, inliers);
    std::cout << "T rows is " << t.rows << "\n";
    //cv::Mat_<double> rowToAppend = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    //cv::vconcat(t, rowToAppend, t);
    std::cout << "Transformation matrix is " << t << "\n";
    std::cout << "Type of transformation matrix is " << t.type() << "\n";   
    auto numInliers = cv::countNonZero(inliers);
    double scale;
    Matx33d rotation;
    ShoColumnVector3d translation;
    std::cout << "Number of inliers is " << numInliers << "\n";
    if (numInliers > 0) {
        Mat a = t.colRange(0, 3);
        Mat b = t.colRange(3, 4);
        std::cout << "a is " << a << '\n';
        std::cout << "b is " << b << '\n';
        Mat a_;
        cv::pow(a, (1, 0 / a.rows - 1), a_);
        std::cout << "Type of a_ is " << a_.type() << "\n";
        std::cout << "Size of a_ is " << a_.size() << "\n";
        scale = cv::determinant(a_);
        rotation = Mat (a / scale);
        translation = b;
        auto temp = *this;
        temp.applySimilarity(scale, rotation, translation);
        auto r = rec;

        for (auto&[id, shot] : temp.getReconstructionShots()) {
            r.getReconstructionShots()[id] = shot;
        }

        for (auto&[track, cp] : temp.getCloudPoints()) {
            r.addCloudPoint(cp);
        }

        *this = r;
    }
}

void Reconstruction::applySimilarity(double s, Matx33d a, ShoColumnVector3d b)
{
    for (auto &[trackId, cp] : cloudPoints) {
        const auto pointCoordinates = Mat(convertVecToRowVector(cp.getPosition()));
        ShoColumnVector3d alignedCoordinate = (s *a) * (pointCoordinates);
        alignedCoordinate += b;
        cp.setPosition(Point3d{ alignedCoordinate(0,0), alignedCoordinate(1,0), alignedCoordinate(2,0) });
    }

    for (auto &[shotId, shot] : shots) {
        const auto r = shot.getPose().getRotationMatrix();
        const auto t = shot.getPose().getTranslation();
        const auto rp = r * a.t();
        const auto tp = -rp * b + s * t;
        shot.getPose().setRotationVector(Mat(rp));
        shot.getPose().setTranslation(tp);
    }
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

