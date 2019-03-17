#include "reconstructor.h"
#include <boost/graph/adjacency_iterator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <algorithm>
#include "multiview.h"
#include "transformations.h"
using namespace cv;

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "bundle.h"
#include "utilities.h"


using csfm::TriangulateBearingsMidpoint;
using std::cerr;
using std::cout;
using std::map;
using std::pair;
using std::set;
using std::sort;
using std::tuple;
using std::vector;
using std::cerr;
using std::string;
using std::endl;
using std::max_element;
using std::make_tuple;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::RowVector3d;
using Eigen::Dynamic;
using Eigen::Map;
using cv::cv2eigen;

Reconstructor ::Reconstructor(
    FlightSession flight, TrackGraph tg,
    std::map<string, TrackGraph::vertex_descriptor> trackNodes,
    std::map<string, TrackGraph::vertex_descriptor> imageNodes)
    : flight(flight),
      tg(tg),
      trackNodes(trackNodes),
      imageNodes(imageNodes),
      shotOrigins(),
      rInverses() {}

void Reconstructor::_alignMatchingPoints(const CommonTrack track,
                                         vector<Point2f>& points1,
                                         vector<Point2f>& points2) const {
  const auto im1 = getImageNode(track.imagePair.first);
  const auto im2 = getImageNode(track.imagePair.second);

  const auto tracks = track.commonTracks;
  map<string, Point2f> aPoints1, aPoints2;
  const auto [edges1Begin, edges1End] = boost::out_edges(im1, this->tg);
  const auto [edges2Begin, edges2End] = boost::out_edges(im2, this->tg);

  for (auto edgeIt = edges1Begin; edgeIt != edges1End; ++edgeIt) {
    if (tracks.find(this->tg[*edgeIt].trackName) != tracks.end()) {
      aPoints1[this->tg[*edgeIt].trackName] =
          this->tg[*edgeIt].fProp.coordinates;
    }
  }

  for (auto edgeIt = edges2Begin; edgeIt != edges2End; ++edgeIt) {
    if (tracks.find(this->tg[*edgeIt].trackName) != tracks.end()) {
      aPoints2[this->tg[*edgeIt].trackName] =
          this->tg[*edgeIt].fProp.coordinates;
    }
  }

  for (auto track : tracks) {
    points1.push_back(aPoints1[track]);
    points2.push_back(aPoints2[track]);
  }

  assert(points1.size() == tracks.size() && points2.size() == tracks.size());
}

vector<DMatch> Reconstructor::_getTrackDMatchesForImagePair(
    const CommonTrack track) const {
  const auto im1 = getImageNode(track.imagePair.first);
  const auto im2 = getImageNode(track.imagePair.second);

  const auto tracks = track.commonTracks;
  map<string, ImageFeatureNode> aPoints1, aPoints2;
  pair<out_edge_iterator, out_edge_iterator> im1Edges =
      boost::out_edges(im1, this->tg);
  pair<out_edge_iterator, out_edge_iterator> im2Edges =
      boost::out_edges(im2, this->tg);
  for (; im1Edges.first != im1Edges.second; ++im1Edges.first) {
    if (tracks.find(this->tg[*im1Edges.first].trackName) != tracks.end()) {
      aPoints1[this->tg[*im1Edges.first].trackName] =
          this->tg[*im1Edges.first].fProp.featureNode;
    }
  }
  for (; im2Edges.first != im2Edges.second; ++im2Edges.first) {
    if (tracks.find(this->tg[*im2Edges.first].trackName) != tracks.end()) {
      aPoints2[this->tg[*im2Edges.first].trackName] =
          this->tg[*im2Edges.first].fProp.featureNode;
    }
  }

  vector<DMatch> imageTrackMatches;
  for (auto track : tracks) {
    imageTrackMatches.push_back(
        {aPoints1[track].second, aPoints2[track].second, 1.0});
  }
  assert(imageTrackMatches.size() == tracks.size());

  return imageTrackMatches;
}

void Reconstructor::_addCameraToBundle(BundleAdjuster &ba,
                                       const Camera camera) {
  cout << "Size of distortion coefficients for this camera is " << camera.getDistortionMatrix().size() << "\n";
  ba.AddPerspectiveCamera("1", camera.getPhysicalFocalLength(), camera.getK1(),
                          camera.getK2(), camera.getInitialPhysicalFocal(),
                          camera.getInitialK1(), camera.getInitialK2(), true);
}

void Reconstructor::_getCameraFromBundle(BundleAdjuster &ba, Camera &cam) {
  auto c = ba.GetPerspectiveCamera("1");
  cam.setFocalWithPhysical(c.GetFocal());
  cam.setK1(c.GetK1());
  cam.setK2(c.GetK2());
}

TwoViewPose Reconstructor::recoverTwoCameraViewPose(CommonTrack track,
                                                    Mat &mask) {
  vector<Point2f> points1;
  vector<Point2f> points2;
  this->_alignMatchingPoints(track, points1, points2);
  auto kMatrix = this->flight.getCamera().getNormalizedKMatrix();
  Mat essentialMatrix = findEssentialMat(points1, points2, kMatrix);
  Mat r, t;
  recoverPose(essentialMatrix, points1, points2, kMatrix, r, t, mask);
  return std::make_tuple(essentialMatrix, r, t);
}

float Reconstructor::computeReconstructabilityScore(int tracks, Mat mask,
                                                    int tresh) {
  auto inliers = countNonZero(mask);
  auto outliers = tracks - inliers;
  auto ratio = float(outliers) / tracks;
  ;
  return ratio > tresh ? ratio : 0;
}

void Reconstructor::computeReconstructability(
    const ShoTracker &tracker, vector<CommonTrack>& commonTracks) {
  auto imageNodes = tracker.getImageNodes();
  for (auto &track : commonTracks) {
    Mat mask;
    this->recoverTwoCameraViewPose(track, mask);
    auto score =
        this->computeReconstructabilityScore(track.commonTracks.size(), mask);
    track.rScore = score;
  }
  sort(std::begin(commonTracks), std::end(commonTracks),
       [](CommonTrack a, CommonTrack b) { return -a.rScore > -b.rScore; });
}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper
//by brown ”
std::tuple<Mat, Mat, Mat, Mat>
Reconstructor::computePlaneHomography(CommonTrack commonTrack) const {
  vector<Point2f> points1;
  vector<Point2f> points2;
  this->_alignMatchingPoints(commonTrack, points1, points2);
  Mat mask;
  auto hom = findHomography(points1, points2, mask, RANSAC,
                                REPROJECTION_ERROR_SD);
  return std::make_tuple(hom, Mat(points1), Mat(points2), mask);
}


void Reconstructor::runIncrementalReconstruction(const ShoTracker& tracker) {
  set<string> reconstructionImages;
  for (const auto it : this->imageNodes) {
    reconstructionImages.insert(it.first);
  }
  auto commonTracks = tracker.commonTracks(this->tg);
  this->computeReconstructability(tracker, commonTracks);
  for (auto track : commonTracks) {
    if (reconstructionImages.find(track.imagePair.first) !=
            reconstructionImages.end() &&
        reconstructionImages.find(track.imagePair.second) !=
            reconstructionImages.end()) {
      cout << "Starting reconstruction with " << track.imagePair.first
           << " and "
           << " and " << track.imagePair.second << '\n';
      auto optRec = beginReconstruction(track, tracker);
      if (optRec) {
        auto rec = *optRec;
        reconstructionImages.erase(track.imagePair.first);
        reconstructionImages.erase(track.imagePair.second);
        continueReconstruction(rec);
      }
    }
  }
}

Reconstructor::OptionalReconstruction Reconstructor::beginReconstruction(CommonTrack track, const ShoTracker &tracker)
{
    Reconstruction rec(flight.getCamera());

    Mat mask;
    TwoViewPose poseParameters = this->recoverTwoCameraViewPose(track, mask);
    Mat essentialMat = std::get<0>(poseParameters);
    Mat r = std::get<1>(poseParameters);
    Mat t = std::get<2>(poseParameters);

    if (essentialMat.rows != 3)
    {
        cout << "Could not compute the essential matrix for this pair" << endl;
        //Get the first essential Mat;
        return rec;
    }

    auto inliers = countNonZero(mask);
    if (inliers <= 5)
    {
        cout << "This pair failed to adequately reconstruct" << endl;
        return std::nullopt;
    }
    Mat rVec;
    Rodrigues(r, rVec);
    Mat distortion;

    const auto shot1Image = flight.getImageSet()[flight.getImageIndex(track.imagePair.first)];
    const auto shot2Image = flight.getImageSet()[flight.getImageIndex(track.imagePair.second)];
    ShotMetadata shot1Metadata(shot1Image.getMetadata(), flight);
    ShotMetadata shot2Metadata(shot2Image.getMetadata(), flight);
    Shot shot1(track.imagePair.first, this->flight.getCamera(), Pose(), shot1Metadata);
    Shot shot2(track.imagePair.second, this->flight.getCamera(), Pose(rVec, t), shot2Metadata);

    rec.getReconstructionShots()[shot1.getId()] = shot1;
    rec.getReconstructionShots()[shot2.getId()] = shot2;

    this->triangulateShots(track.imagePair.first, rec);
    if (rec.getCloudPoints().size() < MIN_INLIERS)
    {
        //return None
        cout << "Initial motion did not generate enough points : " << rec.getCloudPoints().size() << endl;
        return std::nullopt;
  }

  cout << "Generated " << rec.getCloudPoints().size()
       << "points from initial motion " << endl;

  singleViewBundleAdjustment(track.imagePair.second, rec);
  retriangulate(rec);
  singleViewBundleAdjustment(track.imagePair.second, rec);

  return rec;
}

void Reconstructor::continueReconstruction(Reconstruction& rec) {
    bundle(rec);
    removeOutliers(rec);
    alignReconstruction(rec);
}

void Reconstructor::triangulateShots(string image1, Reconstruction &rec) {
  cout << "Triangulating shots " << endl;
  auto im1 = this->imageNodes[image1];

  const auto [edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

  for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
    auto track = this->tg[*tracksIter].trackName;
    cout << "Triangulating track " << track << endl;
    this->triangulateTrack(track, rec);
    cout << "******************************" << endl;
  }
}

void Reconstructor::triangulateTrack(string trackId, Reconstruction& rec) {
  auto track = this->trackNodes[trackId];
  std::pair<adjacency_iterator, adjacency_iterator> neighbors =
      boost::adjacent_vertices(track, this->tg);
  Eigen::Vector3d x;
  vector<Eigen::Vector3d> originList, bearingList;
  for (; neighbors.first != neighbors.second; ++neighbors.first) {
    auto shotId = this->tg[*neighbors.first].name;
    cout << "shot id is " << shotId << endl;
    if (rec.hasShot(shotId)) {
      auto shot = rec.getReconstructionShots()[shotId];
      // cout << "Currently at shot " << shot.getId() << endl;
      auto edgePair = boost::edge(track, this->imageNodes[shotId], this->tg);
      auto edgeDescriptor = edgePair.first;
      auto fCol = this->tg[edgeDescriptor].fProp.color;
      auto fPoint = this->tg[edgeDescriptor].fProp.coordinates;
      auto fBearing =
          this->flight.getCamera().normalizedPointToBearingVec(fPoint);
      // cout << "F point to f bearing is " << fPoint << " to " << fBearing <<
      // endl;
      auto origin = this->getShotOrigin(shot);
      // cout << "Origin for this shot was " << origin << endl;
      Eigen::Vector3d eOrigin;
      Eigen::Matrix3d eigenRotationInverse;
      cv2eigen(Mat(origin), eOrigin);
      auto rotationInverse = this->getRotationInverse(shot);
      cv2eigen(rotationInverse, eigenRotationInverse);
      // cout << "Rotation inverse is " << eigenRotationInverse << endl;
      auto eigenRotationBearingProduct = eigenRotationInverse * fBearing;
      // cout << "Rotation inverse times bearing us  " <<
      // eigenRotationBearingProduct << endl;
      bearingList.push_back(eigenRotationBearingProduct);
      originList.push_back(eOrigin);
    }
  }
  if (bearingList.size() >= 2) {
    if (TriangulateBearingsMidpoint(originList, bearingList, x)) {
      cout << "Triangulation occured succesfully" << endl;
      CloudPoint cp;
      cp.setId(stoi(trackId));
      cp.setPosition(Point3d{x(0), x(1), x(2)});
      rec.addCloudPoint(cp);
    }
  }
}

void Reconstructor::retriangulate(Reconstruction& rec) {
  set<string> tracks;
  for (const auto [imageName, shot] : rec.getReconstructionShots()) {
    try {
      const auto imageVertex = imageNodes.at(imageName);
      const auto [edgesBegin, edgesEnd] =
          boost::out_edges(imageVertex, this->tg);

      for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
        const auto trackName = this->tg[*edgesIter].trackName;
        tracks.insert(trackName);
      }
    } catch (std::out_of_range &e) {
      std::cerr << imageName << "is not valid for this reconstrucion \n";
    }
  }

  for (const auto track : tracks) {
    triangulateTrack(track, rec);
  }
}

ShoColumnVector3d Reconstructor::getShotOrigin(const Shot& shot) {
  auto shotId = shot.getId();
  if (this->shotOrigins.find(shotId) == this->shotOrigins.end()) {
    this->shotOrigins[shotId] = shot.getPose().getOrigin();
  }
  return this->shotOrigins[shotId];
}

Mat Reconstructor::getRotationInverse(const Shot& shot) {
  auto shotId = shot.getId();
  if (this->rInverses.find(shotId) == this->rInverses.end()) {
    auto rotationInverse = shot.getPose().getRotationMatrixInverse();
    this->rInverses[shotId] = rotationInverse;
  }
  return this->rInverses[shotId];
}

void Reconstructor::singleViewBundleAdjustment(std::string shotId,
                                               Reconstruction &rec) {
  BundleAdjuster bundleAdjuster;
  auto shot = rec.getReconstructionShots()[shotId];
  auto camera = shot.getCamera();

  _addCameraToBundle(bundleAdjuster, camera);

  const auto r = shot.getPose().getRotationVector();
  const auto t = shot.getPose().getTranslation();

  bundleAdjuster.AddShot(shot.getId(), "1", r(0), r(1),
                         r(2), t(0), t(1),
                         t(2), false);

  auto im1 = this->imageNodes[shotId];

  const auto [edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

  for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
    auto trackId = this->tg[*tracksIter].trackName;
    try {
      const auto track = rec.getCloudPoints().at(stoi(trackId));
      const auto p = track.getPosition();
      const auto featureCoords = this->tg[*tracksIter].fProp.coordinates;
      bundleAdjuster.AddPoint(trackId, p.x, p.y, p.z, true);
      bundleAdjuster.AddObservation(shotId, trackId, featureCoords.x,
                                    featureCoords.y);
    } catch (std::out_of_range &e) {
      // Pass
    }
  }

  bundleAdjuster.SetLossFunction(LOSS_FUNCTION, LOSS_FUNCTION_TRESHOLD);
  bundleAdjuster.SetReprojectionErrorSD(REPROJECTION_ERROR_SD);
  bundleAdjuster.SetInternalParametersPriorSD(
      EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
      RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
      RADIAL_DISTORTION_K3_SD);
  bundleAdjuster.SetNumThreads(NUM_PROCESESS);
  bundleAdjuster.SetMaxNumIterations(MAX_ITERATIONS);
  bundleAdjuster.SetLinearSolverType(LINEAR_SOLVER_TYPE);
  bundleAdjuster.Run();

  cerr << bundleAdjuster.BriefReport() << "\n";

  auto s = bundleAdjuster.GetShot(shotId);
  Mat rotation = (Mat_<double>(3, 1) << s.GetRX(), s.GetRY(), s.GetRZ());
  Mat translation = (Mat_<double>(3, 1) << s.GetTX(), s.GetTY(), s.GetTZ());
  shot.getPose().setRotationVector(rotation);
  shot.getPose().setTranslation(translation);
}

const vertex_descriptor Reconstructor::getImageNode(string imageName) const {
  return imageNodes.at(imageName);
}

const vertex_descriptor Reconstructor::getTrackNode(string trackId) const {
  return trackNodes.at(trackId);
}

void Reconstructor::plotTracks(CommonTrack track) const {
  Mat imageMatches;
  Mat image1 = imread(
      (this->flight.getImageDirectoryPath() / track.imagePair.first).string(),
      IMREAD_GRAYSCALE);
  Mat image2 = imread(
      (this->flight.getImageDirectoryPath() / track.imagePair.second).string(),
      IMREAD_GRAYSCALE);

  const vertex_descriptor im1 = this->getImageNode(track.imagePair.first);
  const vertex_descriptor im2 = this->getImageNode(track.imagePair.second);
  const auto im1Feats = this->flight.loadFeatures(track.imagePair.first);
  const auto im2Feats = this->flight.loadFeatures(track.imagePair.second);

  auto kp1 = im1Feats.getKeypoints();
  auto kp2 = im2Feats.getKeypoints();

  for (auto &kp : kp1) {
    kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
  }

  for (auto &kp : kp2) {
    kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
  }

  const auto dMatches = _getTrackDMatchesForImagePair(track);

  drawMatches(image1, kp1, image2, kp2, dMatches, imageMatches, Scalar::all(-1),
              Scalar::all(-1), vector<char>(),
              DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  const auto frameName =
      track.imagePair.first + " - " + track.imagePair.second + " tracks";
  namedWindow(frameName, WINDOW_NORMAL);
  imshow(frameName, imageMatches);
}

void Reconstructor::bundle(Reconstruction& rec) {
  BundleAdjuster bundleAdjuster;

  _addCameraToBundle(bundleAdjuster, rec.getCamera());

  for (const auto [shotId, shot] : rec.getReconstructionShots()) {
    const auto r = shot.getPose().getRotationVector();
    const auto t = shot.getPose().getTranslation();


    bundleAdjuster.AddShot(shot.getId(), "1", r(0), r(1),
                           r(2), t(0), t(1), t(2), false);
  }

  for (const auto [id, cloudPoint] : rec.getCloudPoints()) {
    const auto coord = cloudPoint.getPosition();
    bundleAdjuster.AddPoint(std::to_string(id), coord.x, coord.y, coord.z,
                            false);
  }

  for (const auto [shotId, shot] : rec.getReconstructionShots()) {
    try {
      const auto imageVertex = imageNodes.at(shotId);
      const auto [edgesBegin, edgesEnd] =
          boost::out_edges(imageVertex, this->tg);

      for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
        const auto trackName = this->tg[*edgesIter].trackName;
        if (rec.getCloudPoints().find(stoi(trackName)) !=
            rec.getCloudPoints().end()) {
          const auto featureCoords = this->tg[*edgesIter].fProp.coordinates;
          bundleAdjuster.AddObservation(shotId, trackName, featureCoords.x,
                                        featureCoords.y);
        }
      }
    } catch (std::out_of_range &e) {
      cerr << shotId << " Was not found in this reconstruction " << endl;
    }
  }

  bundleAdjuster.SetLossFunction(LOSS_FUNCTION, LOSS_FUNCTION_TRESHOLD);
  bundleAdjuster.SetReprojectionErrorSD(REPROJECTION_ERROR_SD);
  bundleAdjuster.SetInternalParametersPriorSD(
      EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
      RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
      RADIAL_DISTORTION_K3_SD);
  bundleAdjuster.SetNumThreads(NUM_PROCESESS);
  bundleAdjuster.SetMaxNumIterations(50);
  bundleAdjuster.SetLinearSolverType("SPARCE_SCHUR");
  bundleAdjuster.Run();

  _getCameraFromBundle(bundleAdjuster, rec.getCamera());

  for (auto [shotId, shot] : rec.getReconstructionShots()) {
    auto s = bundleAdjuster.GetShot(shotId);
    Mat rotation = (Mat_<double>(3, 1) << s.GetRX(), s.GetRY(), s.GetRZ());
    Mat translation = (Mat_<double>(3, 1) << s.GetTX(), s.GetTY(), s.GetTZ());
    shot.getPose().setRotationVector(rotation);
    shot.getPose().setTranslation(translation);
  }

  for (auto [pointId, cloudPoint] : rec.getCloudPoints()) {
    auto point = bundleAdjuster.GetPoint(std::to_string(pointId));
    cloudPoint.getPosition().x = point.GetX();
    cloudPoint.getPosition().y = point.GetY();
    cloudPoint.getPosition().z = point.GetZ();
    cloudPoint.setError(point.reprojection_error);
  }
}

vector<pair<string, int>> Reconstructor::reconstructedPointForImages(const Reconstruction & rec)
{
    vector<pair <string, int>> res;
    for (const auto[imageName, imageNode] : this->imageNodes) {
        if (!rec.hasShot(imageName)) {
            auto commonTracks = 0;
            const auto[edgesBegin, edgesEnd] = boost::out_edges(imageNode, this->tg);
            for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
                const auto trackName = this->tg[*tracksIter].trackName;
                if (rec.hasTrack(trackName)) {
                    commonTracks++;
                }
            }
            res.push_back(std::make_pair(imageName, commonTracks));
        }
        std::sort(res.begin(), res.end(), [](pair<string, int> a, pair<string, int> b) {
            return a.second < b.second;
        });
    }
    return res;
}

tuple<double, Mat3d, Vec3d> Reconstructor::alignReconstruction(Reconstruction & rec)
{
    cout << "Aligning reconstruction \n";
    double s;
    Mat shotOrigins;
    Mat a,b;
    vector <Point3d> gpsPositions;
    Mat plane;
    Mat verticals;

    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        shotOrigins.push_back(shot.getPose().getOrigin());
        gpsPositions.push_back(shot.getMetadata().gpsPosition);
        const auto[x, y, z] = shot.getOrientationVectors();

        // We always assume that the orientation type is always horizontal
        plane.push_back(x);
        plane.push_back(z);
        verticals.push_back(-y);
    }
    vector<Vec3d> shotOriginsRowMean;
    reduce(shotOrigins, shotOriginsRowMean, 0, REDUCE_AVG);
    auto p = fitPlane(shotOriginsRowMean, plane, verticals);
    auto rPlane = calculateHorizontalPlanePosition(Mat(p));
    Mat3d cvRPlane;
    eigen2cv(rPlane, cvRPlane);
    Mat dotPlaneProduct = (shotOrigins.t() * cvRPlane).t();
    const auto shotOriginStds = getStdByAxis(shotOrigins, 0);
    const auto maxOriginStdIt = max_element(shotOriginStds.begin(), shotOriginStds.end());
    double maxOriginStd = shotOriginStds[std::distance(shotOriginStds.begin(), maxOriginStdIt)];

    const auto gpsPositionStds = getStdByAxis(gpsPositions, 0);
    const auto gpsStdIt = max_element(gpsPositionStds.begin(), gpsPositionStds.end());
    double maxGpsPositionStd = gpsPositionStds[std::distance(shotOriginStds.begin(), gpsStdIt)];
    if (dotPlaneProduct.rows < 2 || maxOriginStd < 1e-8){
        s = dotPlaneProduct.rows / max(1e-8, maxOriginStd);
        a = cvRPlane;
        
        const auto originMeans = getMeanByAxis(shotOrigins, 0);
        const auto gpsMeans = getMeanByAxis(gpsPositions, 0);
        b = Mat(gpsMeans) - Mat(originMeans);
        return make_tuple(s, a, b);
    }
    else {
        auto tAffine = getAffine2dMatrixNoShearing(shotOrigins, gpsPositions);
        //TODO apply scalar operation to s
        const auto s = pow(determinant(tAffine), 0.5);
        auto a = Mat3d::eye(3,3);
        tAffine = tAffine / s;
        cv::Mat aBlock = a(cv::Rect(0, 0, 2, 2));
        tAffine.copyTo(aBlock);
        a *= cvRPlane;
        Mat_<float> b(3, 1);
        b.at<float>(0, 0) = tAffine.at<float>(0, 2);
        b.at<float>(1, 0) = tAffine.at<float>(1, 2);
        b.at<float>(2, 0) = mean(shotOrigins.colRange(0, 2))[0] - mean(s * Mat(gpsPositions).reshape(1).colRange(0, 2))[0];
        return make_tuple(s, a, b);
    }
}

bool Reconstructor::shouldBundle(const Reconstruction &rec)
{
    auto static numPointsLast = rec.getCloudPoints().size();
    auto numShotsLast = rec.getReconstructionShots().size();
    auto interval = 999999;
    auto newPointsRatio = 1.2;

    auto maxPoints = numPointsLast * newPointsRatio;
    auto maxShots = numShotsLast + interval;

    return rec.getCloudPoints().size() >= maxPoints ||
        rec.getReconstructionShots().size() >= maxShots;
}

bool Reconstructor::shouldTriangulate()
{
    return false;
}

void Reconstructor::removeOutliers(Reconstruction& rec) {
  erase_if(
        rec.getCloudPoints(),
        []( const auto& cp ) {
            const auto& [ id, p] = cp;
            return p.getError() > BUNDLE_OUTLIER_THRESHOLD;
        }
    );
}

tuple<bool, ReconstructionReport> Reconstructor::resect(Reconstruction & rec, const vertex_descriptor imageVertex, double threshold,
    int iterations, double probability, int resectionInliers) {
    ReconstructionReport report;
    opengv::points_t Xs;
    opengv::bearingVectors_t Bs;
    const auto[edgesBegin, edgesEnd] = boost::out_edges(imageVertex, this->tg);
    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        const auto trackName = this->tg[*tracksIter].trackName;
        if (rec.hasTrack(trackName)) {
            auto fPoint = this->tg[*tracksIter].fProp.coordinates;
            auto fBearing =
                this->flight.getCamera().normalizedPointToBearingVec(fPoint);
            auto position = rec.getCloudPoints()[stoi(trackName)].getPosition();
            Xs.push_back({position.x, position.y, position.z});
            Bs.push_back(fBearing);
        }
        if (Bs.size() < 5) {
            report.numCommonPoints = Bs.size();
            return make_tuple(false, report);
        }
       
        const auto T = absolutePoseRansac(Bs, Xs, threshold, iterations, probability);
        Matrix3d rotation;
        RowVector3d translation;

        rotation = T.leftCols(3);
        translation = T.rightCols(1).transpose();

        auto fd = Xs.data()->data();
        auto bd = Bs.data()->data();
        Matrix<double, Dynamic, 3> eigenXs = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(fd, Xs.size(), 3);
        Matrix<double, Dynamic, 3> eigenBs = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bd, Bs.size(), 3);
        const auto reprojectedBs = (rotation.transpose() * (eigenXs.rowwise() - translation).transpose()).transpose().matrix();

        /*
        reprojectedBs.colwise() /= reprojectedBs.colwise().norm();

        const auto inliers = ((reprojectedBs.rowwise() - eigenBs).colwise().norm() < threshold).count();
        report.numCommonPoints = Bs.size();
        report.numInliers = inliers;

        if (inliers > resectionInliers) {
            const auto shotName = tg[imageVertex].name;
            const auto shotImageIndex = flight.getImageIndex(shotName);
            const auto shotImage = flight.getImageSet()[shotImageIndex];
            ShotMetadata shotMetadata(shotImage.getMetadata(), flight);
            //TODO add rotation and translation to this pose 
            Shot shot(shotName, flight.getCamera(), Pose(), shotMetadata);
            rec.getReconstructionShots()[shot.getId()] = shot;
            return make_tuple(true, report);
        }
        return make_tuple(false, report);
          */
    }
    return make_tuple(false, report);
}