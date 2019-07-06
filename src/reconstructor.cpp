#include "reconstructor.h"
#include <boost/graph/adjacency_iterator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <algorithm>
#include "multiview.h"
#include "transformations.h"
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>  
#include <opengv/relative_pose/methods.hpp>
#include <opengv/triangulation/methods.hpp>
#include "OpenMVSExporter.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "bundle/bundle_adjuster.h"
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
using std::to_string;
using std::endl;
using std::make_tuple;
using cv::DMatch;
using cv::Point2d;
using cv::Point2f;
using cv::Point_;
using cv::Mat;
using cv::Matx33d;
using cv::Vec2d;
using cv::imread;
using cv::IMREAD_GRAYSCALE;
using cv::Mat3d;
using cv::Point3d;
using cv::Scalar;
using cv::Mat_;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::RowVector3d;
using Eigen::Dynamic;
using Eigen::Map;
using cv::cv2eigen;
using opengv::bearingVectors_t;
using opengv::relative_pose::CentralRelativeAdapter;
using opengv::point_t;
using opengv::triangulation::triangulate;
using opengv::rotation_t;
using opengv::translation_t;

Reconstructor::Reconstructor(
    FlightSession flight, TrackGraph tg,
    std::map<string, TrackGraph::vertex_descriptor> trackNodes,
    std::map<string, TrackGraph::vertex_descriptor> imageNodes)
    : flight_(flight),
    tg_(tg),
    trackNodes_(trackNodes),
    imageNodes_(imageNodes),
    shotOrigins(),
    rInverses() {}

void Reconstructor::_alignMatchingPoints(const CommonTrack track,
    vector<Point2f>& points1,
    vector<Point2f>& points2) const {
    const auto im1 = getImageNode(track.imagePair.first);
    const auto im2 = getImageNode(track.imagePair.second);

    const auto tracks = track.commonTracks;
    map<string, Point2d> aPoints1, aPoints2;
    const auto[edges1Begin, edges1End] = boost::out_edges(im1, tg_);
    const auto[edges2Begin, edges2End] = boost::out_edges(im2, tg_);

    for (auto edgeIt = edges1Begin; edgeIt != edges1End; ++edgeIt) {
        if (tracks.find(this->tg_[*edgeIt].trackName) != tracks.end()) {
            aPoints1[this->tg_[*edgeIt].trackName] =
                this->tg_[*edgeIt].fProp.coordinates;
        }
    }

    for (auto edgeIt = edges2Begin; edgeIt != edges2End; ++edgeIt) {
        if (tracks.find(this->tg_[*edgeIt].trackName) != tracks.end()) {
            aPoints2[this->tg_[*edgeIt].trackName] =
                this->tg_[*edgeIt].fProp.coordinates;
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
        boost::out_edges(im1, this->tg_);
    pair<out_edge_iterator, out_edge_iterator> im2Edges =
        boost::out_edges(im2, this->tg_);
    for (; im1Edges.first != im1Edges.second; ++im1Edges.first) {
        if (tracks.find(tg_[*im1Edges.first].trackName) != tracks.end()) {
            aPoints1[tg_[*im1Edges.first].trackName] =
                tg_[*im1Edges.first].fProp.featureNode;
        }
    }
    for (; im2Edges.first != im2Edges.second; ++im2Edges.first) {
        if (tracks.find(this->tg_[*im2Edges.first].trackName) != tracks.end()) {
            aPoints2[this->tg_[*im2Edges.first].trackName] =
                this->tg_[*im2Edges.first].fProp.featureNode;
        }
    }

    vector<DMatch> imageTrackMatches;
    for (auto track : tracks) {
        imageTrackMatches.push_back(
            { aPoints1[track].second, aPoints2[track].second, 1.0 });
    }
    assert(imageTrackMatches.size() == tracks.size());

    return imageTrackMatches;
}

void Reconstructor::_addCameraToBundle(BundleAdjuster &ba,
    const Camera camera, bool fixCameras) {
    ba.AddPerspectiveCamera("1", camera.getPhysicalFocalLength(), camera.getK1(),
        camera.getK2(), camera.getInitialPhysicalFocal(),
        camera.getInitialK1(), camera.getInitialK2(), fixCameras);
}

void Reconstructor::_getCameraFromBundle(BundleAdjuster &ba, Camera &cam) {
    auto c = ba.GetPerspectiveCamera("1");
    cam.setFocalWithPhysical(c.GetFocal());
    cam.setK1(c.GetK1());
    cam.setK2(c.GetK2());
}

TwoViewPose Reconstructor::recoverTwoCameraViewPose(CommonTrack track, Mat &mask) {
    vector<Point2f> points1;
    vector<Point2f> points2;
    _alignMatchingPoints(track, points1, points2);
    const auto kMatrix = flight_.getCamera().getNormalizedKMatrix();
    Mat essentialMatrix = findEssentialMat(points1, points2, kMatrix);

    if (essentialMatrix.rows == 12 && essentialMatrix.cols == 3) {
        //Special case we have multiple solutions.
        essentialMatrix = essentialMatrix.rowRange(0, 3);
    }
    else if (essentialMatrix.rows != 3 || essentialMatrix.cols != 3) {
        return std::make_tuple(false, Mat(), Mat(), Mat());
    }

    Mat r, t;
    recoverPose(essentialMatrix, points1, points2, kMatrix, r, t, mask);

    return std::make_tuple(true, essentialMatrix, r, t);
}

TwoViewPose Reconstructor::twoViewReconstructionRotationOnly(CommonTrack track, cv::Mat & mask)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    _alignMatchingPoints(track, points1, points2);

    auto bearings1 = flight_.getCamera().normalizedPointsToBearingVec(points1);
    auto bearings2 = flight_.getCamera().normalizedPointsToBearingVec(points2);

    CentralRelativeAdapter adapter(bearings1, bearings2);
    size_t iterations = 100;
    rotation_t relativeRotation;
    for (auto i = 0; i < iterations; ++i) {
        relativeRotation = opengv::relative_pose::rotationOnly(adapter);
    }
    return  _computeRotationInliers(bearings1, bearings2, relativeRotation, mask);
}

template <typename T>
void Reconstructor::twoViewReconstructionInliers(vector<Mat>& Rs_decomp, vector<Mat>& ts_decomp, vector<int> possibleSolutions,
    vector<Point_<T>> points1, vector<Point_<T>> points2) const
{

    auto bearings1 = flight_.getCamera().normalizedPointsToBearingVec(points1);
    auto bearings2 = flight_.getCamera().normalizedPointsToBearingVec(points2);

    for (auto solution : possibleSolutions) {
        auto r = Rs_decomp[solution];
        cout << " r is " << r << "\n";
        rotation_t rotation;
        translation_t translation;
        cv2eigen(r.t(), rotation);
        auto t = ts_decomp[solution];
        cout << "t is " << t << "\n";
        cv2eigen(t, translation);
        _computeTwoViewReconstructionInliers(bearings1, bearings2, rotation, translation.transpose());
    }
}

TwoViewPose Reconstructor::recoverTwoViewPoseWithHomography(CommonTrack track, Mat& mask)
{
    const auto&[hom, points1, points2, homMask] = computePlaneHomography(track);
    homMask.copyTo(mask);
    //cout << "Homography was " << hom << endl;
    if (!hom.rows || !hom.cols)
        return { false, Mat(), Mat(), Mat() };
    vector<Mat> rsDecomp, tsDecomp, normals_decomp;
    Mat gHom = flight_.getCamera().getNormalizedKMatrix() * hom * flight_.getCamera().getNormalizedKMatrix().inv();
    gHom /= gHom.at<double>(2, 2);
    int solutions = decomposeHomographyMat(gHom, flight_.getCamera().getNormalizedKMatrix(), rsDecomp, tsDecomp, normals_decomp);
    vector<int> filteredSolutions;
    cv::filterHomographyDecompByVisibleRefpoints(rsDecomp, normals_decomp, points1, points2, filteredSolutions, homMask);
    if (filteredSolutions.size() > 0) {
        return { true, hom, rsDecomp[filteredSolutions[0]], tsDecomp[filteredSolutions[0]] };
    }
    else if (filteredSolutions.size() > 1) {
        twoViewReconstructionInliers(rsDecomp, tsDecomp, filteredSolutions, points1, points2);
    }

    else {
        cout << "No filtered solutions << \n";
    }



    return { false, Mat(), Mat(), Mat() };
}

void Reconstructor::_computeTwoViewReconstructionInliers(opengv::bearingVectors_t b1, opengv::bearingVectors_t b2,
    opengv::rotation_t r, opengv::translation_t t) const
{
#if 1
    CentralRelativeAdapter adapter(b1, b2, t, r);
    // run method 
    cout << "Number of adapter correspondences is " << adapter.getNumberCorrespondences();

    size_t iterations = 100;
    MatrixXd triangulate_results(3, adapter.getNumberCorrespondences());
    for (size_t i = 0; i < adapter.getNumberCorrespondences(); ++i) {
        for (size_t j = 0; j < iterations; j++)
            triangulate_results.block<3, 1>(0, i) = triangulate(adapter, i);

    }
    MatrixXd error(1, adapter.getNumberCorrespondences());
    for (size_t i = 0; i < adapter.getNumberCorrespondences(); i++)
    {
        Vector3d singleError = triangulate_results.col(i) - b1[i];
        error(0, i) = singleError.norm();
    }
    cout << "Triangulation error is " << error << "\n";
#endif
}

TwoViewPose Reconstructor::_computeRotationInliers(opengv::bearingVectors_t& b1, opengv::bearingVectors_t& b2,
    const opengv::rotation_t& rotation, Mat& cvMask) const
{
    const Eigen::Matrix<double, Dynamic, 3> bearings2 = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(b2.data()->data(),
        b2.size(), 3);
    const Eigen::Matrix<double, Dynamic, 3> bearings1 = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(b1.data()->data(),
        b1.size(), 3);

    auto threshold = 4 * REPROJECTION_ERROR_SD;
    MatrixXd bearingsRotated2(bearings1.rows(), bearings1.cols());
    bearingsRotated2 = (rotation * (bearings2.transpose())).transpose();
    MatrixXd  bearingsdiff(bearings1.rows(), bearings1.cols());
    bearingsdiff = (bearingsRotated2 - bearings1);
    Eigen::MatrixXd maskDiff(bearings1.rows(), 1);
    Eigen::MatrixXi mask(bearings1.rows(), 1);
    maskDiff = bearingsdiff.rowwise().norm();
    mask = (maskDiff.array() < threshold).cast<int>();
    eigen2cv(mask, cvMask);
    Mat rotationCv;
    eigen2cv(rotation, rotationCv);
    return { true, rotationCv, Mat(), Mat() };
}

float Reconstructor::computeReconstructabilityScore(int tracks, Mat mask,
    int tresh) {
    //We use rotation only corrspondence to compute the reconstruction score
    auto inliers = countNonZero(mask);
    auto outliers = tracks - inliers;
    auto ratio = float(inliers) / tracks;
    if (1.0 - ratio >= 0.3)
        return outliers;
    else
        return 0;
}

void Reconstructor::computeReconstructability(
    const ShoTracker &tracker, vector<CommonTrack>& commonTracks) {
    auto imageNodes = tracker.getImageNodes();
    for (auto &track : commonTracks) {
        Mat mask;
        bool success{};
        Mat essentialMat, rotation, translation;
        std::tie(success, essentialMat, rotation, translation) = twoViewReconstructionRotationOnly(track, mask);
        if (success) {
            track.rScore = computeReconstructabilityScore(track.commonTracks.size(), mask);
        }
        else {
            track.rScore = 0.0;
        }
    }

    sort(std::begin(commonTracks), std::end(commonTracks),
        [](const CommonTrack& a, const CommonTrack& b) { return a.rScore > b.rScore; });

}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper
//by brown ”
std::tuple<Mat, vector<Point2f>, vector<Point2f>, Mat>
Reconstructor::computePlaneHomography(CommonTrack commonTrack) const {
    vector<Point2f> points1;
    vector<Point2f> points2;
    _alignMatchingPoints(commonTrack, points1, points2);
    Mat mask;
    auto hom = findHomography(points1, points2, mask, cv::RANSAC,
        REPROJECTION_ERROR_SD);
    return make_tuple(hom, points1, points2, mask);
}


void Reconstructor::runIncrementalReconstruction(const ShoTracker& tracker) {
    //undistort all images 
    //flight.undistort();
    vector<Reconstruction> reconstructions;
    set<string> reconstructionImages;
    for (const auto it : this->imageNodes_) {
        reconstructionImages.insert(it.first);
    }
    auto commonTracks = tracker.commonTracks(this->tg_);
    computeReconstructability(tracker, commonTracks);
    for (auto track : commonTracks) {
        if (reconstructionImages.find(track.imagePair.first) !=
            reconstructionImages.end() &&
            reconstructionImages.find(track.imagePair.second) !=
            reconstructionImages.end()) {

            cerr << "Starting reconstruction with " << track.imagePair.first
                << " and "
                << " and " << track.imagePair.second << '\n';
            auto optRec = beginReconstruction(track, tracker);
            if (optRec) {
                auto rec = *optRec;
                reconstructionImages.erase(track.imagePair.first);
                reconstructionImages.erase(track.imagePair.second);
                continueReconstruction(rec, reconstructionImages);
                string recFileName = flight_.getImageDirectoryPath().parent_path().leaf().string() + "-" +
                    to_string(reconstructions.size() + 1) + ".ply";

                string mvsFileName = flight_.getImageDirectoryPath().parent_path().leaf().string() + "-" +
                    to_string(reconstructions.size() + 1) + ".mvs";
                colorReconstruction(rec);
                rec.saveReconstruction(recFileName);
                exportToMvs(rec, mvsFileName);
                reconstructions.push_back(rec);
            }
        }
    }
    cerr << "Generated a total of " << reconstructions.size() << " partial reconstruction \n";
    if (reconstructions.size() > 1) {
        //We have multiple partial reconstructions. Try to merge all of them
        reconstructions[0].mergeReconstruction(reconstructions[1]);
        string mergedRec = flight_.getImageDirectoryPath().parent_path().leaf().string() + "merged.ply";
        reconstructions[0].alignToGps();
        reconstructions[0].saveReconstruction(mergedRec);
    }
    auto totalPoints = 0;
    for (auto & rec : reconstructions) {
        totalPoints += rec.getCloudPoints().size();
    }
    cerr << "Total number of points in all reconstructions is " << totalPoints << "\n";
}

Reconstructor::OptionalReconstruction Reconstructor::beginReconstruction(CommonTrack track, const ShoTracker &tracker)
{
    Reconstruction rec(flight_.getCamera());

    //Disable gps alignment. Alignment is broken
    rec.setGPS(flight_.hasGps());
    Mat mask;
    //TwoViewPose poseParameters = recoverTwoCameraViewPose(track, mask);
    TwoViewPose poseParameters = recoverTwoViewPoseWithHomography(track, mask);

    auto[success, poseMatrix, rotation, translation] = poseParameters;
    Mat essentialMat = std::get<1>(poseParameters);
    Mat r = std::get<2>(poseParameters);
    Mat t = std::get<3>(poseParameters);

    if (poseMatrix.rows != 3 || !success)
    {
        cout << "Could not compute the essential matrix for this pair" << endl;
        //Get the first essential Mat;
        return std::nullopt;
    }

    auto inliers = countNonZero(mask);
    if (inliers <= 5)
    {
        cerr << "This pair failed to adequately reconstruct" << endl;
        return std::nullopt;
    }

    Mat rVec;
    Rodrigues(r, rVec);
    Mat distortion;

    const auto shot1Image = flight_.getImageSet()[flight_.getImageIndex(track.imagePair.first)];
    const auto shot2Image = flight_.getImageSet()[flight_.getImageIndex(track.imagePair.second)];
    ShotMetadata shot1Metadata(shot1Image.getMetadata(), flight_);
    ShotMetadata shot2Metadata(shot2Image.getMetadata(), flight_);
    Shot shot1(track.imagePair.first, flight_.getCamera(), Pose(), shot1Metadata);
    Shot shot2(track.imagePair.second, flight_.getCamera(), Pose(rVec, t), shot2Metadata);

    rec.addShot(shot1.getId(), shot1);
    rec.addShot(shot2.getId(), shot2);

    triangulateShotTracks(track.imagePair.first, rec);
    if (rec.getCloudPoints().size() < MIN_INLIERS)
    {
        //return None
        cerr << "Initial motion did not generate enough points : " << rec.getCloudPoints().size() << endl;
        return std::nullopt;
    }
    colorReconstruction(rec);
    rec.saveReconstruction("green.ply");
    cerr << "Generated " << rec.getCloudPoints().size() << "points from initial motion " << endl;
    singleViewBundleAdjustment(track.imagePair.second, rec);
    retriangulate(rec);
    singleViewBundleAdjustment(track.imagePair.second, rec);
    return rec;
}

void Reconstructor::continueReconstruction(Reconstruction& rec, set<string>& images) {
    bundle(rec);
    removeOutliers(rec);
    rec.alignToGps();
    colorReconstruction(rec);
    rec.saveReconstruction("partialgreen.ply");
    rec.updateLastCounts();
    while (1) {
        auto candidates = reconstructedPointForImages(rec, images);
        if (candidates.empty())
            break;

        for (auto[imageName, numTracks] : candidates) {
            auto before = rec.getCloudPoints().size();
            auto imageVertex = getImageNode(imageName);
            auto[status, report] = resect(rec, imageVertex);
            if (!status)
                continue;

            singleViewBundleAdjustment(imageName, rec);
            rec.saveReconstruction("partialgreen.ply");
            cerr << "Adding " << imageName << " to the reconstruction \n";
            images.erase(imageName);
            triangulateShotTracks(imageName, rec);

            if (rec.needsRetriangulation()) {
                cerr << "Retriangulating reconstruction \n";
                bundle(rec);
                retriangulate(rec);
                bundle(rec);
                removeOutliers(rec);
                rec.alignToGps();
                rec.updateLastCounts();
            }
            else if (rec.needsBundling()) {
                bundle(rec);
                removeOutliers(rec);
                rec.alignToGps();
                rec.updateLastCounts();
            }
            else {
                //TODO implement local neighbourhood shot bundling
            }
            auto after = rec.getCloudPoints().size();
            if (after - before > 0 && before > 0)
            {
                cerr << "Added " << after - before << " points to the reconstruction \n";
            }
        }
        bundle(rec);
        removeOutliers(rec);
        rec.alignToGps();
        return;
    }
}

void Reconstructor::triangulateShotTracks(string image1, Reconstruction &rec) {
    cout << "Triangulating tracks for " << image1 << '\n';
    auto im1 = imageNodes_[image1];

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg_);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto track = tg_[*tracksIter].trackName;
        if (rec.getCloudPoints().find(stoi(track)) == rec.getCloudPoints().end())
            triangulateTrack(track, rec);
    }
}

void Reconstructor::triangulateTrack(string trackId, Reconstruction& rec) {
    rInverses.clear();
    shotOrigins.clear();
    auto track = trackNodes_[trackId];
    std::pair<adjacency_iterator, adjacency_iterator> neighbors =
        boost::adjacent_vertices(track, this->tg_);
    Eigen::Vector3d x;
    vector<Eigen::Vector3d> originList, bearingList;
    for (; neighbors.first != neighbors.second; ++neighbors.first) {
        auto shotId = this->tg_[*neighbors.first].name;
        if (rec.hasShot(shotId)) {
            auto shot = rec.getShot(shotId);
            auto edgePair = boost::edge(track, this->imageNodes_[shotId], this->tg_);
            auto edgeDescriptor = edgePair.first;
            auto fCol = tg_[edgeDescriptor].fProp.color;
            auto fPoint = tg_[edgeDescriptor].fProp.coordinates;
            auto fBearing = flight_.getCamera().normalizedPointToBearingVec(fPoint);
            auto origin = getShotOrigin(shot);
            Eigen::Vector3d eOrigin;
            Eigen::Matrix3d eigenRotationInverse;
            cv2eigen(Mat(origin), eOrigin);
            auto rotationInverse = getRotationInverse(shot);
            cv2eigen(rotationInverse, eigenRotationInverse);
            auto eigenRotationBearingProduct = eigenRotationInverse * fBearing;
            bearingList.push_back(eigenRotationBearingProduct);
            originList.push_back(eOrigin);
        }
    }
    if (bearingList.size() >= 2) {
        if (TriangulateBearingsMidpoint(originList, bearingList, x)) {
            CloudPoint cp;
            cp.setId(stoi(trackId));
            cp.setPosition(Point3d{ x(0), x(1), x(2) });
            rec.addCloudPoint(cp);
        }
    }
}

void Reconstructor::retriangulate(Reconstruction& rec) {
    rInverses.clear();
    shotOrigins.clear();
    set<string> tracks;
    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes_.at(imageName);
            const auto[edgesBegin, edgesEnd] =
                boost::out_edges(imageVertex, this->tg_);

            for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
                const auto trackName = this->tg_[*edgesIter].trackName;
                tracks.insert(trackName);
            }
        }
        catch (std::out_of_range &e) {
            std::cerr << imageName << "is not valid for this reconstrucion \n";
        }
    }

    for (const auto track : tracks) {
        triangulateTrack(track, rec);
    }
}

ShoColumnVector3d Reconstructor::getShotOrigin(const Shot& shot) {
    auto shotId = shot.getId();
    if (shotOrigins.find(shotId) == shotOrigins.end()) {
        shotOrigins[shotId] = shot.getPose().getOrigin();
    }
    return shotOrigins[shotId];
}

Mat Reconstructor::getRotationInverse(const Shot& shot) {
    auto shotId = shot.getId();
    if (rInverses.find(shotId) == rInverses.end()) {
        auto rotationInverse = shot.getPose().getRotationMatrixInverse();
        rInverses[shotId] = rotationInverse;
    }
    return rInverses[shotId];
}

void Reconstructor::singleViewBundleAdjustment(std::string shotId,
    Reconstruction &rec) {
    BundleAdjuster bundleAdjuster;
    auto shot = rec.getShot(shotId);
    auto camera = shot.getCamera();

    _addCameraToBundle(bundleAdjuster, camera, OPTIMIZE_CAMERA_PARAEMETERS);

    const auto r = shot.getPose().getRotationVector();
    const auto t = shot.getPose().getTranslation();

    bundleAdjuster.AddShot(
        shot.getId(), "1",
        { r(0), r(1),r(2) },
        { t(0), t(1),t(2) },
        false
    );


    auto im1 = imageNodes_[shotId];

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg_);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto trackId = tg_[*tracksIter].trackName;
        try {
            const auto track = rec.getCloudPoints().at(stoi(trackId));
            const auto p = track.getPosition();
            const auto featureCoords = tg_[*tracksIter].fProp.coordinates;
            bundleAdjuster.AddPoint(trackId, { p.x, p.y, p.z }, true);
            bundleAdjuster.AddPointProjectionObservation(
                shotId,
                trackId,
                featureCoords.x,
                featureCoords.y,
                tg_[*tracksIter].fProp.scale
            );
        }
        catch (std::out_of_range &e) {
            // Pass
        }
    }

    if (flight_.hasGps() && rec.usesGps()) {
        cout << "Using gps prior \n";
        const auto g = shot.getMetadata().gpsPosition;
        bundleAdjuster.AddPositionPrior(shotId, g.x, g.y, g.z, shot.getMetadata().gpsDop);
    }

    bundleAdjuster.SetPointProjectionLossFunction(LOSS_FUNCTION, LOSS_FUNCTION_TRESHOLD);
    bundleAdjuster.SetInternalParametersPriorSD(
        EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
        RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
        RADIAL_DISTORTION_K3_SD);
    bundleAdjuster.SetNumThreads(NUM_PROCESESS);
    bundleAdjuster.SetMaxNumIterations(MAX_ITERATIONS);
    bundleAdjuster.SetLinearSolverType(LINEAR_SOLVER_TYPE);
    bundleAdjuster.Run();

    cerr << bundleAdjuster.FullReport() << "\n";

    auto s = bundleAdjuster.GetShot(shotId);
    Mat rotation = (Mat_<double>(3, 1) << s.GetRotation().x(), s.GetRotation().y(), s.GetRotation().z());
    auto translation = s.GetTranslation();

    shot.getPose().setRotationVector(rotation);
    shot.getPose().setTranslation({ translation.x(), translation.y(), translation.z() });

    rec.addShot(shotId, shot);
}

const vertex_descriptor Reconstructor::getImageNode(string imageName) const {
    return imageNodes_.at(imageName);
}

const vertex_descriptor Reconstructor::getTrackNode(string trackId) const {
    return trackNodes_.at(trackId);
}

void Reconstructor::plotTracks(CommonTrack track) const {
    Mat imageMatches;
    Mat image1 = imread((flight_.getImageDirectoryPath() / track.imagePair.first).string(), IMREAD_GRAYSCALE);
    Mat image2 = imread((flight_.getImageDirectoryPath() / track.imagePair.second).string(), IMREAD_GRAYSCALE);

    const vertex_descriptor im1 = getImageNode(track.imagePair.first);
    const vertex_descriptor im2 = getImageNode(track.imagePair.second);
    const auto im1Feats = flight_.loadFeatures(track.imagePair.first);
    const auto im2Feats = flight_.loadFeatures(track.imagePair.second);

    auto kp1 = im1Feats.getKeypoints();
    auto kp2 = im2Feats.getKeypoints();

    for (auto &kp : kp1) {
        kp.pt = flight_.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    for (auto &kp : kp2) {
        kp.pt = flight_.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    const auto dMatches = _getTrackDMatchesForImagePair(track);

    drawMatches(image1, kp1, image2, kp2, dMatches, imageMatches, Scalar::all(-1),
        Scalar::all(-1), vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const auto frameName = track.imagePair.first + " - " + track.imagePair.second + " tracks";
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
}

void Reconstructor::exportToMvs(const Reconstruction & rec, const std::string mvsFileName)
{
    csfm::OpenMVSExporter exporter;
    exporter.AddCamera("1", rec.getCamera().getNormalizedKMatrix());

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        auto imagePath = flight_.getUndistortedImagesDirectoryPath() / shotId;
        imagePath.replace_extension("png");
        auto origin = Mat(shot.getPose().getOrigin()).data;
        exporter.AddShot(
            imagePath.string(),
            shotId,
            "1",
            shot.getPose().getRotationMatrix(),
            { static_cast<double>(origin[0]), static_cast<double>(origin[1]), static_cast<double>(origin[2]) }
        );
    }

    for (const auto &[trackId, cloudPoint] : rec.getCloudPoints()) {
        vector<string> shots;
        auto trackVertex = trackNodes_.at(std::to_string(trackId));

        const auto[edgesBegin, edgesEnd] =
            boost::out_edges(trackVertex, tg_);

        for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
            const auto shotName = tg_[*edgesIter].imageName;
            shots.push_back(shotName);
        }
        exporter.AddPoint(cloudPoint.getPosition(), shots);
    }
    exporter.Export(mvsFileName);
}

template <class> struct TD;

void Reconstructor::bundle(Reconstruction& rec) {
    auto fixCameras = !OPTIMIZE_CAMERA_PARAEMETERS;
    BundleAdjuster bundleAdjuster;

    _addCameraToBundle(bundleAdjuster, rec.getCamera(), fixCameras);

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        const auto r = shot.getPose().getRotationVector();
        const auto t = shot.getPose().getTranslation();


        bundleAdjuster.AddShot(shot.getId(), "1", { r(0), r(1),
            r(2) }, { t(0), t(1), t(2) }, false);
    }

    for (const auto[id, cloudPoint] : rec.getCloudPoints()) {
        const auto coord = cloudPoint.getPosition();
        bundleAdjuster.AddPoint(std::to_string(id), { coord.x, coord.y, coord.z },
            false);
    }

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes_.at(shotId);
            const auto[edgesBegin, edgesEnd] =
                boost::out_edges(imageVertex, this->tg_);

            for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
                const auto trackName = this->tg_[*edgesIter].trackName;
                if (rec.getCloudPoints().find(stoi(trackName)) !=
                    rec.getCloudPoints().end()) {
                    const auto featureCoords = this->tg_[*edgesIter].fProp.coordinates;
                    bundleAdjuster.AddPointProjectionObservation(shotId, trackName, featureCoords.x,
                        featureCoords.y, tg_[*edgesIter].fProp.scale);
                }
            }
        }
        catch (std::out_of_range &e) {
            cerr << shotId << " Was not found in this reconstruction " << endl;
        }
    }

    if (flight_.hasGps() && rec.usesGps()) {
        cout << "Using gps prior \n";
        for (const auto[shotId, shot] : rec.getReconstructionShots()) {
            const auto g = shot.getMetadata().gpsPosition;
            bundleAdjuster.AddPositionPrior(shotId, g.x, g.y, g.z, shot.getMetadata().gpsDop);
        }
    }

    bundleAdjuster.SetInternalParametersPriorSD(
        EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
        RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
        RADIAL_DISTORTION_K3_SD);
    bundleAdjuster.SetNumThreads(NUM_PROCESESS);
    bundleAdjuster.SetMaxNumIterations(50);
    bundleAdjuster.SetLinearSolverType("SPARSE_SCHUR");
    bundleAdjuster.Run();

    _getCameraFromBundle(bundleAdjuster, rec.getCamera());

    for (auto&[shotId, shot] : rec.getReconstructionShots()) {
        auto s = bundleAdjuster.GetShot(shotId);
        Mat rotation = (Mat_<double>(3, 1) << s.GetRotation().x(), s.GetRotation().y(), s.GetRotation().z());
        auto translation = s.GetTranslation();

        shot.getPose().setRotationVector(rotation);
        shot.getPose().setTranslation({ translation.x(), translation.y(), translation.z() });
    }

    for (auto &[pointId, cloudPoint] : rec.getCloudPoints()) {
        auto point = bundleAdjuster.GetPoint(std::to_string(pointId));
        auto triangulatedPosition = point.GetPoint();
        cloudPoint.getPosition().x = triangulatedPosition.x();
        cloudPoint.getPosition().y = triangulatedPosition.y();
        cloudPoint.getPosition().z = triangulatedPosition.z();
        cloudPoint.setError(point.reprojection_errors);
    }
}

vector<pair<string, int>> Reconstructor::reconstructedPointForImages(const Reconstruction & rec, set<string> & images)
{
    vector<pair <string, int>> res;
    for (const auto imageName : images) {
        if (!rec.hasShot(imageName)) {
            auto commonTracks = 0;
            const auto[edgesBegin, edgesEnd] = boost::out_edges(getImageNode(imageName), tg_);
            for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
                const auto trackName = this->tg_[*tracksIter].trackName;
                if (rec.hasTrack(trackName)) {
                    commonTracks++;
                }
            }
            res.push_back(std::make_pair(imageName, commonTracks));
        }
        std::sort(res.begin(), res.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second > b.second;
        });
    }
    return res;
}

void Reconstructor::colorReconstruction(Reconstruction & rec)
{
    for (auto&[trackId, cp] : rec.getCloudPoints()) {
        const auto trackNode = getTrackNode(to_string(trackId));
        auto[edgesBegin, _] = boost::out_edges(trackNode, this->tg_);
        edgesBegin++;
        cp.setColor(tg_[*edgesBegin].fProp.color);
    }
}

void Reconstructor::removeOutliers(Reconstruction& rec) {
    const auto before = rec.getCloudPoints().size();
    vector<pair<string, string>>  outliers;

    for (const auto &[track, cloudPoint] : rec.getCloudPoints()) {
        for (const auto&[shotId, errors] : cloudPoint.getError()) {
            if (errors.norm() > BUNDLE_OUTLIER_THRESHOLD) {
                outliers.push_back(make_pair(to_string(track), shotId));
            }
        }
    }

    for (const auto &[track, shotId] : outliers) {
        if (!rec.hasTrack(track))
            continue;
        rec.getCloudPoints().at(stoi(track)).getError().erase(shotId);
        auto iter = tg_.out_edge_list(trackNodes_.at(track)).begin();
        boost::remove_edge(trackNodes_.at(track), imageNodes_.at(shotId), tg_);
        //tg_.remove_edge(boost::edge(trackNodes_[track], imageNodes_[shotId], tg_).first);
    }

    for (const auto &[track, _] : outliers) {
        auto numEdges = boost::out_degree(trackNodes_[track], tg_);
        if (numEdges < 2) {
            rec.getCloudPoints().erase(stoi(track));
            boost::remove_vertex(trackNodes_[track], tg_);
            trackNodes_.erase(track);
        }
    }

#if 0
    erase_if(
        rec.getCloudPoints(),
        [](const auto& cp) {
        const auto&[id, p] = cp;
        return (p.getError() > BUNDLE_OUTLIER_THRESHOLD);
    }
    );
#endif
    cerr << "Removed " << outliers.size() << " outliers from reconstruction \n";
}

tuple<bool, ReconstructionReport> Reconstructor::resect(Reconstruction & rec, const vertex_descriptor imageVertex, double threshold,
    int iterations, double probability, int resectionInliers) {
    ReconstructionReport report;
    vector<Point2d> fPoints;
    vector<Point3d> realWorldPoints;
    const auto[edgesBegin, edgesEnd] = boost::out_edges(imageVertex, this->tg_);
    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        const auto trackName = tg_[*tracksIter].trackName;
        if (rec.hasTrack(trackName)) {
            auto fPoint = tg_[*tracksIter].fProp.coordinates;
            auto fBearing = flight_.getCamera().normalizedPointToBearingVec(fPoint);
            fPoints.push_back(flight_.getCamera().denormalizeImageCoordinates(fPoint));
            auto position = rec.getCloudPoints().at(stoi(trackName)).getPosition();
            realWorldPoints.push_back(position);
        }
    }

    if (realWorldPoints.size() < 5) {
        report.numCommonPoints = realWorldPoints.size();
        return make_tuple(false, report);
    }

    Mat pnpRot, pnpTrans, inliers;
    if (cv::solvePnPRansac(realWorldPoints, fPoints, flight_.getCamera().getKMatrix(),
        flight_.getCamera().getDistortionMatrix(), pnpRot, pnpTrans, false, iterations, 8.0, probability, inliers)) {
        const auto shotName = tg_[imageVertex].name;
        const auto shot = flight_.getImageSet()[flight_.getImageIndex(shotName)];
        ShotMetadata shotMetadata(shot.getMetadata(), flight_);
        report.numCommonPoints = realWorldPoints.size();
        report.numInliers = cv::countNonZero(inliers);
        Shot recShot(shotName, flight_.getCamera(), Pose(pnpRot, pnpTrans), shotMetadata);
        rec.addShot(recShot.getId(), recShot);
        return { true, report };
    }
    return make_tuple(false, report);
}