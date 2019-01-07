#include "reconstructor.h"
#include "multiview.h"
#include <vector>
#include <boost/graph/adjacency_iterator.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>


using std::cout;
using std::map;
using std::pair;
using std::set;
using std::sort;
using std::vector;
using csfm::TriangulateBearingsMidpoint;

Reconstructor ::Reconstructor(FlightSession flight, TrackGraph tg, std::map<string, TrackGraph::vertex_descriptor> trackNodes,
                              std::map<string, TrackGraph::vertex_descriptor> imageNodes) : flight(flight), tg(tg), 
                              trackNodes(trackNodes), imageNodes(imageNodes), shotOrigins(), rInverses(){}

void Reconstructor::_alignMatchingPoints(void *img1, void *img2, const set<string> &tracks, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2)
{
    Mat cameraMatrix = (Mat_<double>(3, 3) << 3.8123526712521689e+3, 0.2592, 0, 3.8123526712521689e+03, 1944, 0, 0.1);
    map<string, Point2f> aPoints1, aPoints2;
    pair<out_edge_iterator, out_edge_iterator> im1Edges = boost::out_edges(img1, this->tg);
    pair<out_edge_iterator, out_edge_iterator> im2Edges = boost::out_edges(img2, this->tg);
    for (; im1Edges.first != im1Edges.second; ++im1Edges.first)
    {
        if (tracks.find(this->tg[*im1Edges.first].trackName) != tracks.end())
        {
            aPoints1[this->tg[*im1Edges.first].trackName] = this->tg[*im1Edges.first].fProp.coordinates;
        }
    }
    for (; im2Edges.first != im2Edges.second; ++im2Edges.first)
    {
        if (tracks.find(this->tg[*im2Edges.first].trackName) != tracks.end())
        {
            aPoints2[this->tg[*im2Edges.first].trackName] = this->tg[*im2Edges.first].fProp.coordinates;
        }
    }

    for (auto track : tracks)
    {
        points1.push_back(aPoints1[track]);
        points2.push_back(aPoints2[track]);
    }
    assert(points1.size() == tracks.size() && points2.size() == tracks.size());
}

void Reconstructor::recoverTwoCameraViewPose(void *image1, void *image2, std::set<string> tracks, Mat &mask, int method, double tresh, double prob)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(image1, image2, tracks, points1, points2);
    auto kMatrix = this->flight.getCamera().getNormalizedKMatrix();
    Mat essentialMatrix = cv::findEssentialMat(points1, points2, kMatrix, method, tresh, prob, mask);
}

float Reconstructor::computeReconstructabilityScore(int tracks, Mat mask, int tresh)
{
    auto inliers = countNonZero(mask);
    auto outliers = tracks - inliers;
    auto ratio = float(outliers) / tracks;
    ;
    return ratio > tresh ? ratio : 0;
}

void Reconstructor::computeReconstructability(const ShoTracker &tracker, vector<CommonTrack> &commonTracks)
{
    auto imageNodes = tracker.getImageNodes();
    for (auto &track : commonTracks)
    {
        Mat mask;
        auto im1 = imageNodes[track.imagePair.first];
        auto im2 = imageNodes[track.imagePair.second];
        this->recoverTwoCameraViewPose(im1, im2, track.commonTracks, mask);
        auto score = this->computeReconstructabilityScore(track.commonTracks.size(), mask);
        track.rScore = score;
    }
    sort(std::begin(commonTracks), std::end(commonTracks), [](CommonTrack a, CommonTrack b) { return -a.rScore > -b.rScore; });
}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper by brown ”
void Reconstructor::computePlaneHomography(string image1, string image2)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    //this->_alignMatchingPoints(image1, image2, points1, points2);
    // auto h = cv::findHomography(points1, points2);

    //Decompose the recovered homography
}

void Reconstructor::runIncrementalReconstruction(const ShoTracker &tracker)
{
    auto imageNodes = tracker.getImageNodes();
    set<string> reconstructionImages;
    for (auto it : imageNodes)
    {
        reconstructionImages.insert(it.first);
    }
    auto commonTracks = tracker.commonTracks(this->tg);
    this->computeReconstructability(tracker, commonTracks);
    for (auto track : commonTracks)
    {
        if (reconstructionImages.find(track.imagePair.first) != reconstructionImages.end()
         && reconstructionImages.find(track.imagePair.second) != reconstructionImages.end())
        {
            Reconstruction rec = this->beginReconstruction(track.imagePair.first, track.imagePair.second, track.commonTracks, tracker);
        }
        cout << "Pair " << track.imagePair.first << " - " << track.imagePair.second << " has " << track.rScore << " score " << endl;
    }
}

Reconstruction Reconstructor::beginReconstruction(string image1, string image2, set<string> tracks, const ShoTracker &tracker)
{
    Reconstruction rec;
    vector<Point2f> points1;
    vector<Point2f> points2;
    auto im1 = imageNodes[image1];
    auto im2 = imageNodes[image2];
    this->_alignMatchingPoints(im1, im2, tracks, points1, points2);
    auto kMatrix = this->flight.getCamera().getNormalizedKMatrix();
    Mat r, t, mask, essentialMatrix = cv::findEssentialMat(points1, points2, kMatrix);

    if (essentialMatrix.rows != 3) {
        cout << "Could not compute the essential matrix for this pair" << endl;
        //Get the first essential Mat;
        return rec;
    }
    //Decompose the essential matrix
    cv::recoverPose(essentialMatrix, points1, points2, kMatrix, r, t, mask);

    auto inliers = countNonZero(mask);
    if (inliers <= 5)
    {
        cout << "This pair failed to adequately reconstruct" << endl;
    }
    cv::Mat rVec;
    cv::Rodrigues(r, rVec);
    cv::Mat distortion;
    Reconstruction reconstruction;
    Shot shot1(image1, this->flight.getCamera(), Pose());
    Shot shot2(image2, this->flight.getCamera(), Pose(rVec, t));
    cout << "Shot 2 pose is " << shot2.getPose() << endl;
    rec.getReconstructionShots()[shot1.getId()] = shot1;
    rec.getReconstructionShots()[shot2.getId()] = shot2;
    this->triangulateShots(image1, rec);
    return rec;
}

void Reconstructor::triangulateShots(string image1, Reconstruction &rec)
{
    cout << "Triangulating shots "<<endl;
    auto im1 = this->imageNodes[image1];
    pair<out_edge_iterator, out_edge_iterator> im1Edges = boost::out_edges(im1, this->tg);
    for (; im1Edges.first != im1Edges.second; ++im1Edges.first)
    {
        auto track = this->tg[*im1Edges.first].trackName;
        cout << "Triangulating track "<< track << endl;
        this->triangulateTrack(track, rec);
    }
}

void Reconstructor::triangulateTrack(string trackId, Reconstruction& rec)
{
    auto track = this->trackNodes[trackId];
    std::pair<adjacency_iterator, adjacency_iterator> neighbors = boost::adjacent_vertices(track, this->tg);
    Eigen::Vector3d x;
    vector<Eigen::Vector3d> a, b;
    for (; neighbors.first != neighbors.second; ++neighbors.first)
    {
        auto shotId =  this->tg[*neighbors.first].name;
        if (rec.hasShot(shotId)) {
            auto shot = rec.getReconstructionShots()[shotId];
            auto edgePair = boost::edge(track, this->imageNodes[shotId], this->tg);
            auto edgeDescriptor = edgePair.first;
            auto fCol = this->tg[edgeDescriptor].fProp.color;
            auto fPoint = this->tg[edgeDescriptor].fProp.coordinates;
            auto fBearing = this->flight.getCamera().normalizedPointToBearingVec(fPoint);
            cout << "F point to f bearing is " << fPoint << " to " << fBearing << endl;
            auto origin = this->getShotOrigin(shot);
            cout << "Currently at shot " << shot.getId() << endl;
            cout << "Origin for this shot was " << origin << endl;
            Eigen::Vector3d eOrigin;
            Eigen::Matrix3d eigenRotationInverse;
            cv2eigen(origin, eOrigin);
            auto rotationInverse = this->getRotationInverse(shot);
            cv2eigen(rotationInverse, eigenRotationInverse);
            cout << "Rotation inverse is " << eigenRotationInverse << endl;
            auto eigenRotationBearingProduct = eigenRotationInverse * fBearing;
            cout << "Rotation inverse times bearing us  " << eigenRotationBearingProduct<< endl;  
            b.push_back(eigenRotationBearingProduct);
            a.push_back(eOrigin);
            if (TriangulateBearingsMidpoint(a,b,x)) {
                cout << "Triangulation occured succesfully" << endl;
            }
        }
    }
    if (b.size() >= 2) {
        if (TriangulateBearingsMidpoint(a,b,x)) {
            cout << "Triangulaiton occured succesfully" << endl;
            }
    }
}

cv::Mat Reconstructor::getShotOrigin(const Shot& shot) {
    auto shotId = shot.getId();
    if(this->shotOrigins.find(shotId) == this->shotOrigins.end()) {
        this->shotOrigins[shotId] = shot.getPose().getOrigin();
    }
    return this->shotOrigins[shotId];
}

cv::Mat Reconstructor::getRotationInverse(const Shot& shot) {
    auto shotId = shot.getId();
    if(this->rInverses.find(shotId) == this->rInverses.end()) {
        auto r = shot.getPose().getRotationMatrix();
        cv::Mat tR;
        cv::transpose(r, tR );
        this->rInverses[shotId] =  tR;
    }
    return this->rInverses[shotId];
}
