#include "reconstruction.h"
#include "reconstructor.h"
#include "utilities.h"
#include "multiview.h"
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>
#include <string>

using std::map;
using std::string;
using std::ofstream;
using std::ios;
using std::tuple;
using std::max_element;
using std::make_tuple;
using std::set;
using std::max;
using std::vector;
using cv::Mat;
using cv::Matx33d;
using cv::eigen2cv;
using cv::Mat3d;
using cv::Vec2d;
using cv::Point3d;

Reconstruction::Reconstruction() : shots(), cloudPoints(), camera(), lastPointCount(), lastShotCount() {}

Reconstruction::Reconstruction(Camera camera) :camera(camera), lastPointCount(), lastShotCount() {}

void Reconstruction::addShot(std::string shotId, Shot shot)
{
    shots[shotId] = shot;
}

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
            r.addShot(id, shot);
        }

        for (auto&[track, cp] : temp.getCloudPoints()) {
            r.addCloudPoint(cp);
        }

        *this = r;
    }
}

void Reconstruction::alignToGps()
{
    if (!usesGPS)
        return;

    const auto[s, a, b] = getGPSTransform();
    applySimilarity(s, a, b);
}

void Reconstruction::applySimilarity(double s, Matx33d a, ShoColumnVector3d b)
{
    for (auto &[trackId, cp] : cloudPoints) {
        //std::cout << "Track id is " << trackId << "\n";
        const auto pointCoordinates = Mat(convertVecToRowVector(cp.getPosition()));
        //std::cout << "Point coordinates are " << pointCoordinates << "\n";
        ShoColumnVector3d alignedCoordinate = (s *a) * (pointCoordinates);
        //std::cout << "Aligned coordinate before adding b is " << alignedCoordinate << "\n";
        alignedCoordinate += b;
        cp.setPosition(Point3d{ alignedCoordinate(0,0), alignedCoordinate(1,0), alignedCoordinate(2,0) });
    }

    for (auto &[shotId, shot] : shots) {
        const auto r = shot.getPose().getRotationMatrix();
        const auto t = shot.getPose().getTranslation();
        const auto rp = r * a.t();
        const auto tp = -rp * b + (s * t);
        shot.getPose().setRotationVector(Mat(rp));
        shot.getPose().setTranslation(tp);
    }
}

void Reconstruction::setGPS(bool useGps)
{
    usesGPS = useGps;
}

const Shot& Reconstruction::getShot(std::string shotId) const
{
    return shots.at(shotId);
}

Shot& Reconstruction::getShot(string shotId) {
    return shots.at(shotId);
}

bool Reconstruction::usesGps() const
{
    return usesGPS;
}


tuple<double, cv::Matx33d, ShoColumnVector3d> Reconstruction::getGPSTransform()
{
    double s;
    Mat shotOrigins(0, 3, CV_64FC1);
    Mat a;
    vector <ShoRowVector3d>gpsPositions;
    Mat gpsPositions2D, shotOrigins2D;
    Mat plane(0, 3, CV_64FC1);
    Mat verticals(0, 3, CV_64FC1);

    for (const auto[imageName, shot] : shots) {
        auto shotOrigin = Mat(shot.getPose().getOrigin());
        shotOrigin = shotOrigin.reshape(1, 1);
        shotOrigins.push_back(shotOrigin);
        Vec2d shotOrigin2D((double*)shotOrigin.colRange(0, 2).data);
        shotOrigins2D.push_back(shotOrigin2D);
        const auto gpsPosition = shot.getMetadata().gpsPosition;
        gpsPositions.push_back({ gpsPosition.x, gpsPosition.y, gpsPosition.z });
        gpsPositions2D.push_back(Vec2d{ gpsPosition.x, gpsPosition.y });
        const auto[x, y, z] = shot.getOrientationVectors();

        // We always assume that the orientation type is horizontal

        //cout << "Size of x is " << Mat(x).size() << "\n\n";
        //cout << "Type of x is " << Mat(x).type() << "\n\n";
        //cout << "Size of plane is " << plane.size() << "\n\n";
        plane.push_back(Mat(x));
        plane.push_back(Mat(z));
        verticals.push_back(-Mat(y));
    }

    Mat shotOriginsRowMean;
    reduce(shotOrigins, shotOriginsRowMean, 0, cv::REDUCE_AVG);
    Mat shotOriginsSub = shotOrigins.clone();
    for (auto i = 0; i < shotOrigins.rows; ++i) {
        shotOriginsSub.row(i) -= shotOriginsRowMean;
    }
    auto p = fitPlane(shotOriginsSub, plane, verticals);
    auto rPlane = calculateHorizontalPlanePosition(Mat(p));
    Mat cvRPlane(3, 3, CV_64FC1);
    eigen2cv(rPlane, cvRPlane);
#if 0
    std::cout << "Size of CV r plane was " << cvRPlane.size() << "\n";
    std::cout << "Size of shot origins is " << shotOrigins.size() << "\n";

    std::cout << "R plane was " << cvRPlane << "\n";
    std::cout << "Shot origins was " << shotOrigins << "\n";
#endif
    const auto shotOriginsTranspose = shotOrigins.t();
    //TODO check this dotplane product
    Mat dotPlaneProduct = (cvRPlane * shotOrigins.t()).t();
    const auto shotOriginStds = getStdByAxis(shotOrigins, 0);
    const auto maxOriginStdIt = max_element(shotOriginStds.begin(), shotOriginStds.end());
    double maxOriginStd = shotOriginStds[std::distance(shotOriginStds.begin(), maxOriginStdIt)];
    const auto gpsPositionStds = getStdByAxis(gpsPositions, 0);
    const auto gpsStdIt = max_element(gpsPositionStds.begin(), gpsPositionStds.end());
    double maxGpsPositionStd = gpsPositionStds[std::distance(gpsPositionStds.begin(), gpsStdIt)];
    if (dotPlaneProduct.rows < 2 || maxOriginStd < 1e-8) {
        s = dotPlaneProduct.rows / max(1e-8, maxOriginStd);
        a = cvRPlane;

        const auto originMeans = getMeanByAxis(shotOrigins, 0);
        const auto gpsMeans = getMeanByAxis(gpsPositions, 0);
        Mat b = Mat(gpsMeans) - Mat(originMeans);
        return make_tuple(s, a, b);
    }
    else {
        Mat tAffine(3, 3, CV_64FC1);
        //Using input array a vector of Mat type elements with 3 channels stacks them horizontally 
        //auto tAffine = getAffine2dMatrixNoShearing(shotOrigins, Mat(gpsPositions));
       // auto tAffine = getAffine2dMatrixNoShearing(shotOrigins, shotOrigins);
        cv::Mat dotPlane2d = dotPlaneProduct.colRange(0, 2).reshape(2).clone();
        auto checkVector = dotPlane2d.checkVector(2);
        tAffine = estimateAffinePartial2D(dotPlane2d, gpsPositions2D);

        tAffine.push_back(Mat(ShoRowVector3d{ 0,0,1 }));
        //TODO apply scalar operation to s
        const auto s = pow(determinant(tAffine), 0.5);
        auto a = Mat(Matx33d::eye());
        Mat tAffineS = tAffine / s;
        cv::Mat aBlock = a(cv::Rect(0, 0, 2, 2));
        tAffineS(cv::Rect(0, 0, 2, 2)).copyTo(aBlock);
        a *= cvRPlane;
        a *= cvRPlane;
        auto b3 = (mean(Mat(gpsPositions).reshape(1).col(2))[0] - mean(s * dotPlaneProduct.col(2))[0]);
        ShoColumnVector3d b{ tAffine.at<double>(0, 2), tAffine.at<double>(1, 2), b3 };
        return make_tuple(s, a, b);
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

