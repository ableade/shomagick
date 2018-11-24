#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "flightsession.h"
#include "shotracking.h"
#include "camera.h"

class Reconstructor
{
private:
  FlightSession flight;
  TrackGraph tg;
  void _alignMatchingPoints(string image1, string image2, std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);

public:
  Reconstructor(FlightSession flight, TrackGraph tg);
  Reconstructor(FlightSession flight, ShoTracker tracker);
  void recoverTwoCameraViewPose(string image1, string image2, int method = cv::RANSAC, double tresh = 0.999, double prob = 1.0);
  float computeReconstructability(int commonPoints, Mat inliers, int treshold = 0.3);
  void computePlaneHomography(string image1, string image2);
};

#endif