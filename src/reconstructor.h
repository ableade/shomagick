#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "flightsession.h"
#include "shotracking.h"
#include "camera.h"
#include "reconstruction.h"


class Reconstructor
{
private:
  FlightSession flight;
  TrackGraph tg;
  std::map<string, TrackGraph::vertex_descriptor> trackNodes;
  std::map<string, TrackGraph::vertex_descriptor> imageNodes;
  std::map<string, cv::Mat> shotOrigins;
  std::map<string, cv::Mat> rInverses;
  void _alignMatchingPoints(void* img1, void* img2, const std::set<string> &tracks,std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

public:
  Reconstructor(FlightSession flight, TrackGraph tg, std::map<string, TrackGraph::vertex_descriptor> trackNodes, 
  std::map<string, TrackGraph::vertex_descriptor> imageNodes);

  void recoverTwoCameraViewPose(void* image1, void* image2, std::set<string> tracks, cv::Mat& mask, int method = cv::RANSAC, double tresh = 0.999, double prob = 1.0);
  float computeReconstructabilityScore(int tracks, cv::Mat inliers, int treshold = 0.3);
  void computeReconstructability(const ShoTracker& tracker, std::vector<CommonTrack>& commonTracks);
  void computePlaneHomography(string image1, string image2);
  void runIncrementalReconstruction (const ShoTracker& tracker);
  Reconstruction beginReconstruction (string image1, string image2, std::set<string> tracks, const ShoTracker& tracker);
  void triangulateShots(string image1, Reconstruction& rec);
  void triangulateTrack(string trackId, Reconstruction& rec);
  cv::Mat getShotOrigin(const Shot& shot);
  cv::Mat getRotationInverse(const Shot& shot);
};

#endif