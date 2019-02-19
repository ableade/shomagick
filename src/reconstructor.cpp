#include "reconstructor.h"
#include "multiview.h"
#include <vector>
#include <boost/graph/adjacency_iterator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "bundle.h"


using csfm::TriangulateBearingsMidpoint;
using std::cout;
using std::map;
using std::pair;
using std::set;
using std::sort;
using std::vector;
using std::cerr;
using std::string;
using std::endl;

Reconstructor ::Reconstructor(FlightSession flight, TrackGraph tg, std::map<string, TrackGraph::vertex_descriptor> trackNodes,
                              std::map<string, TrackGraph::vertex_descriptor> imageNodes) : flight(flight), tg(tg),
                                                                                            trackNodes(trackNodes), imageNodes(imageNodes), shotOrigins(), rInverses() {}

void Reconstructor::_alignMatchingPoints(const CommonTrack track, vector<cv::Point2f> &points1, vector<cv::Point2f> &points2) const
{
    const auto im1 = getImageNode(track.imagePair.first);
    const auto im2 = getImageNode(track.imagePair.second);

    const auto tracks = track.commonTracks;
    map<string, Point2f> aPoints1, aPoints2;
    pair<out_edge_iterator, out_edge_iterator> im1Edges = boost::out_edges(im1, this->tg);
    pair<out_edge_iterator, out_edge_iterator> im2Edges = boost::out_edges(im2, this->tg);
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

vector<cv::DMatch> Reconstructor::_getTrackDMatchesForImagePair(const CommonTrack track) const
{
    const auto im1 = getImageNode(track.imagePair.first);
    const auto im2 = getImageNode(track.imagePair.second);

    const auto tracks = track.commonTracks;
    map<string, ImageFeatureNode> aPoints1, aPoints2;
    pair<out_edge_iterator, out_edge_iterator> im1Edges = boost::out_edges(im1, this->tg);
    pair<out_edge_iterator, out_edge_iterator> im2Edges = boost::out_edges(im2, this->tg);
    for (; im1Edges.first != im1Edges.second; ++im1Edges.first)
    {
        if (tracks.find(this->tg[*im1Edges.first].trackName) != tracks.end())
        {
            aPoints1[this->tg[*im1Edges.first].trackName] = this->tg[*im1Edges.first].fProp.featureNode;
        }
    }
    for (; im2Edges.first != im2Edges.second; ++im2Edges.first)
    {
        if (tracks.find(this->tg[*im2Edges.first].trackName) != tracks.end())
        {
            aPoints2[this->tg[*im2Edges.first].trackName] = this->tg[*im2Edges.first].fProp.featureNode;
        }
    }

    vector<cv::DMatch> imageTrackMatches;
    for (auto track : tracks)
    {
        imageTrackMatches.push_back({ aPoints1[track].second, aPoints2[track].second, 1.0 });
    }
    assert(imageTrackMatches.size() == tracks.size());

    return imageTrackMatches;
}

void Reconstructor::_addCameraToBundle(BundleAdjuster& ba, const Camera camera) {
    ba.AddPerspectiveCamera("1", camera.getPhysicalFocalLength(),
        camera.getK1(), camera.getK2(), camera.getInitialPhysicalFocal(),
        camera.getInitialK1(), camera.getInitialK2(), true);

}

void Reconstructor::_getCameraFromBundle(BundleAdjuster & ba, Camera & cam)
{
    auto c = ba.GetPerspectiveCamera("1");
    cam.setFocalWithPhysical(c.GetFocal());
    cam.setK1(c.GetK1());
    cam.setK2(c.GetK2());
}

TwoViewPose Reconstructor::recoverTwoCameraViewPose(CommonTrack track, cv::Mat &mask)
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(track, points1, points2);
    auto kMatrix = this->flight.getCamera().getNormalizedKMatrix();
    Mat essentialMatrix = cv::findEssentialMat(points1, points2, kMatrix);
    Mat r, t;
    cv::recoverPose(essentialMatrix, points1, points2, kMatrix, r, t, mask);
    return std::make_tuple(essentialMatrix, r, t);
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
        this->recoverTwoCameraViewPose(track, mask);
        auto score = this->computeReconstructabilityScore(track.commonTracks.size(), mask);
        track.rScore = score;
    }
    sort(std::begin(commonTracks), std::end(commonTracks), [](CommonTrack a, CommonTrack b) { return -a.rScore > -b.rScore; });
}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper by brown ”
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> Reconstructor::computePlaneHomography(CommonTrack commonTrack) const
{
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(commonTrack, points1, points2);
    
   // points1 = this->flight.getCamera().denormalizeImageCoordinates(points1);
    //points2 = this->flight.getCamera().denormalizeImageCoordinates(points2);

    Mat mask;
    auto hom = cv::findHomography(points1, points2, mask, cv::RANSAC, REPROJECTION_ERROR_SD);
    return std::make_tuple( hom, Mat(points1), Mat(points2), mask);
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
        if (reconstructionImages.find(track.imagePair.first) != reconstructionImages.end() && reconstructionImages.find(track.imagePair.second) != reconstructionImages.end())
        {
            cout << "Starting reconstruction with " << track.imagePair.first << " and " <<
                " and " << track.imagePair.second << '\n';
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
    Reconstruction rec;

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
    cv::Mat rVec;
    cv::Rodrigues(r, rVec);
    cv::Mat distortion;
    Reconstruction reconstruction;

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

    cout << "Generated " << rec.getCloudPoints().size() << "points from initial motion " << endl;

    singleViewBundleAdjustment(track.imagePair.second, rec);
    retriangulate(rec);
    singleViewBundleAdjustment(track.imagePair.second, rec);

    return rec;
}

void Reconstructor::continueReconstruction(Reconstruction & rec)
{

}

void Reconstructor::triangulateShots(string image1, Reconstruction &rec)
{
    cout << "Triangulating shots " << endl;
    auto im1 = this->imageNodes[image1];

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto track = this->tg[*tracksIter].trackName;
        cout << "Triangulating track " << track << endl;
        this->triangulateTrack(track, rec);
        cout << "******************************" << endl;
    }
}

void Reconstructor::triangulateTrack(string trackId, Reconstruction &rec)
{
    auto track = this->trackNodes[trackId];
    std::pair<adjacency_iterator, adjacency_iterator> neighbors = boost::adjacent_vertices(track, this->tg);
    Eigen::Vector3d x;
    vector<Eigen::Vector3d> originList, bearingList;
    for (; neighbors.first != neighbors.second; ++neighbors.first)
    {
        auto shotId = this->tg[*neighbors.first].name;
        if (rec.hasShot(shotId))
        {
            auto shot = rec.getReconstructionShots()[shotId];
            //cout << "Currently at shot " << shot.getId() << endl;
            auto edgePair = boost::edge(track, this->imageNodes[shotId], this->tg);
            auto edgeDescriptor = edgePair.first;
            auto fCol = this->tg[edgeDescriptor].fProp.color;
            auto fPoint = this->tg[edgeDescriptor].fProp.coordinates;
            auto fBearing = this->flight.getCamera().normalizedPointToBearingVec(fPoint);
            //cout << "F point to f bearing is " << fPoint << " to " << fBearing << endl;
            auto origin = this->getShotOrigin(shot);
            //cout << "Origin for this shot was " << origin << endl;
            Eigen::Vector3d eOrigin;
            Eigen::Matrix3d eigenRotationInverse;
            cv2eigen(origin, eOrigin);
            auto rotationInverse = this->getRotationInverse(shot);
            cv2eigen(rotationInverse, eigenRotationInverse);
            //cout << "Rotation inverse is " << eigenRotationInverse << endl;
            auto eigenRotationBearingProduct = eigenRotationInverse * fBearing;
            //cout << "Rotation inverse times bearing us  " << eigenRotationBearingProduct << endl;
            bearingList.push_back(eigenRotationBearingProduct);
            originList.push_back(eOrigin);
        }
    }
    if (bearingList.size() >= 2)
    {
        if (TriangulateBearingsMidpoint(originList, bearingList, x))
        {
            cout << "Triangulation occured succesfully" << endl;
            CloudPoint cp;
            cp.setId(stoi(trackId));
            cp.setPosition(Point3d{x(0), x(1), x(2)});
            rec.addCloudPoint(cp);
        }
    }
}

void Reconstructor::retriangulate(Reconstruction & rec)
{
    set<string> tracks;
    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes.at(imageName);
            const auto[edgesBegin, edgesEnd] = boost::out_edges(imageVertex, this->tg);

            for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
                const auto trackName = this->tg[*edgesIter].trackName;
                tracks.insert(trackName);
            }
        }
        catch (std::out_of_range& e) {
            std::cerr << imageName << "is not valid for this reconstrucion \n";
        }
    }

    for (const auto track : tracks) {
        triangulateTrack(track, rec);
    }
}

cv::Mat Reconstructor::getShotOrigin(const Shot &shot)
{
    auto shotId = shot.getId();
    if (this->shotOrigins.find(shotId) == this->shotOrigins.end())
    {
        this->shotOrigins[shotId] = shot.getPose().getOrigin();
    }
    return this->shotOrigins[shotId];
}

cv::Mat Reconstructor::getRotationInverse(const Shot &shot)
{
    auto shotId = shot.getId();
    if (this->rInverses.find(shotId) == this->rInverses.end())
    {
        auto rotationInverse = shot.getPose().getRotationMatrixInverse();
        this->rInverses[shotId] = rotationInverse;
    }
    return this->rInverses[shotId];
}

void Reconstructor::singleViewBundleAdjustment(std::string shotId, Reconstruction &rec)
{
    BundleAdjuster bundleAdjuster;
    auto shot = rec.getReconstructionShots()[shotId];
    auto camera = shot.getCamera();

    _addCameraToBundle(bundleAdjuster, camera);

    const auto r = shot.getPose().getRotationVector();
    const auto t = shot.getPose().getTranslation();

    bundleAdjuster.AddShot(shot.getId(), "1", r.at<double>(0), r.at<double>(1), r.at<double>(2),
        t.at<double>(0), t.at<double>(1), t.at<double>(2),
        false);

    auto im1 = this->imageNodes[shotId];

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto trackId = this->tg[*tracksIter].trackName;
        try {
            const auto track = rec.getCloudPoints().at(stoi(trackId));
            const auto p = track.getPosition();
            const auto featureCoords = this->tg[*tracksIter].fProp.coordinates;
            bundleAdjuster.AddPoint(trackId, p.x, p.y, p.z, true);
            bundleAdjuster.AddObservation(shotId, trackId, featureCoords.x, featureCoords.y);
        }
        catch (std::out_of_range& e) {
            //Pass
        }
    }

    bundleAdjuster.SetLossFunction(LOSS_FUNCTION, LOSS_FUNCTION_TRESHOLD);
    bundleAdjuster.SetReprojectionErrorSD(REPROJECTION_ERROR_SD);
    bundleAdjuster.SetInternalParametersPriorSD(EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
        RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
        RADIAL_DISTORTION_K3_SD);
    bundleAdjuster.SetNumThreads(NUM_PROCESESS);
    bundleAdjuster.SetMaxNumIterations(MAX_ITERATIONS);
    bundleAdjuster.SetLinearSolverType(LINEAR_SOLVER_TYPE);
    bundleAdjuster.Run();

    cerr << bundleAdjuster.BriefReport() << "\n";

    auto s = bundleAdjuster.GetShot(shotId);
    Mat rotation = (cv::Mat_<double>(3, 1) << s.GetRX(), s.GetRY(), s.GetRZ());
    Mat translation = (cv::Mat_<double>(3, 1) << s.GetTX(), s.GetTY(), s.GetTZ());
    shot.getPose().setRotationVector(rotation);
    shot.getPose().setTranslation(translation);
}

const vertex_descriptor Reconstructor::getImageNode(string imageName) const
{
    return imageNodes.at(imageName);
}

const vertex_descriptor Reconstructor::getTrackNode(string trackId) const
{
    return trackNodes.at(trackId);
}

void Reconstructor::plotTracks(CommonTrack track) const
{
    Mat imageMatches;
    Mat image1 = imread((this->flight.getImageDirectoryPath() / track.imagePair.first).string(),
        cv::IMREAD_GRAYSCALE);
    Mat image2 = imread((this->flight.getImageDirectoryPath() / track.imagePair.second).string(),
        cv::IMREAD_GRAYSCALE);

    const vertex_descriptor im1 = this->getImageNode(track.imagePair.first);
    const vertex_descriptor im2 = this->getImageNode(track.imagePair.second);
    const auto im1Feats = this->flight.loadFeatures(track.imagePair.first);
    const auto im2Feats = this->flight.loadFeatures(track.imagePair.second);

    auto kp1 = im1Feats.getKeypoints();
    auto kp2 = im2Feats.getKeypoints();

    for (auto& kp : kp1) {
        kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    for (auto& kp : kp2) {
        kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    const auto dMatches = _getTrackDMatchesForImagePair(track);

    drawMatches(image1, kp1, image2, kp2, dMatches, imageMatches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    const auto frameName = track.imagePair.first + " - " + track.imagePair.second + " tracks";
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
}

void Reconstructor::bundle(Reconstruction & rec)
{
    BundleAdjuster bundleAdjuster;
    _addCameraToBundle(bundleAdjuster, rec.getCamera());

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        const auto r = shot.getPose().getRotationVector();
        const auto t = shot.getPose().getTranslation();

        bundleAdjuster.AddShot(shot.getId(), "1", r.at<double>(0), r.at<double>(1), r.at<double>(2),
            t.at<double>(0), t.at<double>(1), t.at<double>(2),
            false);
    }

    for (const auto[id, cloudPoint] : rec.getCloudPoints()) {
        const auto coord = cloudPoint.getPosition();
        bundleAdjuster.AddPoint(std::to_string(id), coord.x, coord.y, coord.z, false);
    }

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes.at(shotId);
            const auto[edgesBegin, edgesEnd] = boost::out_edges(imageVertex, this->tg);

            for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
                const auto trackName = this->tg[*edgesIter].trackName;
                if (rec.getCloudPoints().find(stoi(trackName)) != rec.getCloudPoints().end()) {
                    const auto featureCoords = this->tg[*edgesIter].fProp.coordinates;
                    bundleAdjuster.AddObservation(shotId, trackName, featureCoords.x, featureCoords.y);
                }
            }
        } catch (std::out_of_range& e) {
            cerr << shotId << " Was not found in this reconstruction " << endl;
        }
    }

    bundleAdjuster.SetLossFunction(LOSS_FUNCTION, LOSS_FUNCTION_TRESHOLD);
    bundleAdjuster.SetReprojectionErrorSD(REPROJECTION_ERROR_SD);
    bundleAdjuster.SetInternalParametersPriorSD(EXIF_FOCAL_SD, PRINCIPAL_POINT_SD, RADIAL_DISTORTION_K1_SD,
        RADIAL_DISTORTION_K2_SD, RADIAL_DISTORTION_P1_SD, RADIAL_DISTORTION_P2_SD,
        RADIAL_DISTORTION_K3_SD);
    bundleAdjuster.SetNumThreads(NUM_PROCESESS);
    bundleAdjuster.SetMaxNumIterations(50);
    bundleAdjuster.SetLinearSolverType("SPARCE_SCHUR");
    bundleAdjuster.Run();

    _getCameraFromBundle(bundleAdjuster, rec.getCamera());

    for (auto[shotId, shot] : rec.getReconstructionShots()) {
        auto s = bundleAdjuster.GetShot(shotId);
        Mat rotation = (cv::Mat_<double>(3, 1) << s.GetRX(), s.GetRY(), s.GetRZ());
        Mat translation = (cv::Mat_<double>(3, 1) << s.GetTX(), s.GetTY(), s.GetTZ());
        shot.getPose().setRotationVector(rotation);
        shot.getPose().setTranslation(translation);
    }

    for (auto[pointId, cloudPoint] : rec.getCloudPoints()) {
        auto point = bundleAdjuster.GetPoint(std::to_string(pointId));
        cloudPoint.getPosition().x = point.GetX();
        cloudPoint.getPosition().y = point.GetY();
        cloudPoint.getPosition().z = point.GetZ();
        cloudPoint.setError(point.reprojection_error);
    }
}

void Reconstructor::removeOutliers(Reconstruction & rec)
{

}

void Reconstructor::alignReconstruction(Reconstruction & rec)
{
    vector<cv::Mat> shotOrigins;
    vector <cv::Point3d> gpsPositions;

    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        shotOrigins.push_back(shot.getPose().getOrigin());
        gpsPositions.push_back(shot.getMetadata().gpsPosition);
        auto rot = shot.getPose().getRotationMatrix();
    }
}

