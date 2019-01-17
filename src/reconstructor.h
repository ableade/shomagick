#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "flightsession.h"
#include "shotracking.h"
#include "camera.h"
#include <tuple>
#include "reconstruction.h"

//Essential matrix, rotation and translation
typedef std::tuple <cv::Mat, cv::Mat, cv::Mat> TwoViewPose;

class Reconstructor
{
private:
  FlightSession flight;
  TrackGraph tg;
  std::map<std::string, TrackGraph::vertex_descriptor> trackNodes;
  std::map<std::string, TrackGraph::vertex_descriptor> imageNodes;
  std::map<std::string, cv::Mat> shotOrigins;
  std::map<std::string, cv::Mat> rInverses;
  void _alignMatchingPoints(void* img1, void* img2, const std::set<string> &tracks,std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

public:
  Reconstructor(FlightSession flight, TrackGraph tg, std::map<string, TrackGraph::vertex_descriptor> trackNodes, 
  std::map<string, TrackGraph::vertex_descriptor> imageNodes);

  TwoViewPose recoverTwoCameraViewPose(void* image1, void* image2, std::set<string> tracks, cv::Mat& mask);
  float computeReconstructabilityScore(int tracks, cv::Mat inliers, int treshold = 0.3);
  void computeReconstructability(const ShoTracker& tracker, std::vector<CommonTrack>& commonTracks);
  void computePlaneHomography(std::string image1, std::string image2);
  void runIncrementalReconstruction (const ShoTracker& tracker);
  Reconstruction beginReconstruction (string image1, string image2, std::set<string> tracks, const ShoTracker& tracker);
  void triangulateShots(std::string image1, Reconstruction& rec);
  void triangulateTrack(std::string trackId, Reconstruction& rec);
  cv::Mat getShotOrigin(const Shot& shot);
  cv::Mat getRotationInverse(const Shot& shot);
  void singleCameraBundleAdjustment(std::string shotId, Reconstruction& rec);
};

#endif