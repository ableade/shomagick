#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "flightsession.h"
#include "shotracking.h"
#include "camera.h"
#include "bundle.h"
#include "reconstruction.h"
#include <tuple>
#include <optional>

//Essential matrix, rotation and translation
typedef std::tuple <cv::Mat, cv::Mat, cv::Mat> TwoViewPose;
//Loss function for the ceres problem (see: http://ceres-solver.org/modeling.html#lossfunction)
const std::string LOSS_FUNCTION = "SoftLOneLoss";
//Threshold on the squared residuals.  Usually cost is quadratic for smaller residuals and sub-quadratic above.
const double LOSS_FUNCTION_TRESHOLD = 1.0;
//The standard deviation of the reprojection error
const double REPROJECTION_ERROR_SD = 0.004;
//The standard deviation of the exif focal length in log - scale
const double EXIF_FOCAL_SD = 0.01;
//The standard deviation of the principal point coordinates
const double PRINCIPAL_POINT_SD = 0.01;
//The standard deviation of the first radial distortion parameter
const double RADIAL_DISTORTION_K1_SD = 0.01;
// The standard deviation of the second radial distortion parameter
const double RADIAL_DISTORTION_K2_SD = 0.01;
//The standard deviation of the first tangential distortion parameter
const double RADIAL_DISTORTION_P1_SD = 0.01;
//The standard deviation of the second tangential distortion parameter
const double RADIAL_DISTORTION_P2_SD = 0.01;
//The standard deviation of the third radial distortion parameter
const double RADIAL_DISTORTION_K3_SD = 0.01;
// Number of threads to use
const int NUM_PROCESESS = 1;
const int MAX_ITERATIONS = 10;
const string LINEAR_SOLVER_TYPE = "DENSE_QR";
const int MIN_INLIERS = 20;
const int BUNDLE_OUTLIER_THRESHOLD = 0.006;

class Reconstructor
{
public:
    using TrackName = std::string;
    using ImageName = std::string;
    using TrackNodes = std::map<TrackName, TrackGraph::vertex_descriptor>;
    using ImageNodes = std::map<ImageName, TrackGraph::vertex_descriptor>;

private:
  FlightSession flight;
  TrackGraph tg;
  TrackNodes trackNodes;
  ImageNodes imageNodes;
  std::map<std::string, cv::Mat> shotOrigins;
  std::map<std::string, cv::Mat> rInverses;
  void _alignMatchingPoints(const CommonTrack track, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2) const;
  std::vector<cv::DMatch> _getTrackDMatchesForImagePair(const CommonTrack track) const;
  void _addCameraToBundle(BundleAdjuster& ba, const Camera camera);
  void _getCameraFromBundle(BundleAdjuster& ba, Camera& cam);

public:
  Reconstructor(FlightSession flight, TrackGraph tg, std::map<string, TrackGraph::vertex_descriptor> trackNodes, 
  std::map<string, TrackGraph::vertex_descriptor> imageNodes);

  TwoViewPose recoverTwoCameraViewPose(CommonTrack track, cv::Mat& mask);
  float computeReconstructabilityScore(int tracks, cv::Mat inliers, int treshold = 0.3);
  void computeReconstructability(const ShoTracker& tracker, std::vector<CommonTrack>& commonTracks);
  std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> computePlaneHomography(CommonTrack commonTrack) const;
  void runIncrementalReconstruction (const ShoTracker& tracker);

  using OptionalReconstruction = std::optional<Reconstruction>;
  OptionalReconstruction beginReconstruction (CommonTrack track, const ShoTracker& tracker);
  void continueReconstruction(Reconstruction& rec);
  void triangulateShots(std::string image1, Reconstruction& rec);
  void triangulateTrack(std::string trackId, Reconstruction& rec);
  void retriangulate(Reconstruction& rec);
  cv::Mat getShotOrigin(const Shot& shot);
  cv::Mat getRotationInverse(const Shot& shot);
  void singleViewBundleAdjustment(std::string shotId, Reconstruction& rec);
  const vertex_descriptor getImageNode(const std::string imageName) const;
  const vertex_descriptor getTrackNode(std::string trackId) const;
  void plotTracks(CommonTrack track) const;
  void bundle(Reconstruction& rec);
  void removeOutliers(Reconstruction & rec);
};

#endif