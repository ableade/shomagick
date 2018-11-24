#include "reconstruction.h"
#include <vector>

using cv::countNonZero;
using cv::Mat;
using cv::Mat_;
using cv::Point2f;
using std::vector;

Reconstructor ::Reconstructor(FlightSession flight, TrackGraph tg) : flight(flight), tg(tg){};

void Reconstructor::_alignMatchingPoints(string image1, string image2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2)
{
    Mat cameraMatrix = (Mat_<double>(3, 3) << 3.8123526712521689e+3, 0.2592, 0, 3.8123526712521689e+03, 1944, 0, 0.1);
    auto allPairMatches = this->flight.loadMatches(image1);
    auto pairwiseMatches = allPairMatches[image2];
    auto features2 = this->flight.loadFeatures(image2);
    auto features1 = this->flight.loadFeatures(image1);

    vector<cv::Point2f> impoints1, impoints2;
    //align pairwise matches
    for (auto dmatch : pairwiseMatches)
    {
        auto point1 = features1.first[dmatch.queryIdx].pt;
        auto point2 = features2.first[dmatch.trainIdx].pt;
        points1.push_back(point1);
        points2.push_back(point2);
    }
}
void Reconstructor::recoverTwoCameraViewPose(string image1, string image2, int method, double tresh, double prob)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(image1, image2, points1, points2);
    Mat cameraMatrix = (Mat_<double>(3, 3) << 3.8123526712521689e+3, 0.2592, 0, 3.8123526712521689e+03, 1944, 0, 0.1);
    Mat mask;
    Mat essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, method, tresh, prob, mask);
}

float Reconstructor::computeReconstructability(int commonPoints, Mat mask, int tresh)
{
    auto inliers = countNonZero(mask);
    auto outliers = commonPoints - inliers;
    auto ratio = float(outliers) / commonPoints;
    return ratio > tresh ? ratio : 0;
}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper v”
void Reconstructor::computePlaneHomography(string image1, string image2)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(image1, image2, points1, points2);
    auto h = cv::findHomography(points1, points2);

    //Decompose the recovered homography
}
