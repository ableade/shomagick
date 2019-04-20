#include "reconstructor.h"
#include <boost/graph/adjacency_iterator.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <algorithm>
#include "multiview.h"
#include "transformations.h"
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/triangulation/methods.hpp>

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
using std::to_string;
using std::endl;
using std::max_element;
using std::make_tuple;
using cv::DMatch;
using cv::Point2d;
using cv::Point2f;
using cv::Mat;
using cv::Matx33d;
using cv::Vec2d;
using cv::imread;
using cv::IMREAD_GRAYSCALE;
using cv::Mat3d;
using cv::Point3d;
using cv::Scalar;
using cv::Mat_;
using cv::eigen2cv;
using std::max;
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
    map<string, Point2d> aPoints1, aPoints2;
    const auto[edges1Begin, edges1End] = boost::out_edges(im1, this->tg);
    const auto[edges2Begin, edges2End] = boost::out_edges(im2, this->tg);

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
            { aPoints1[track].second, aPoints2[track].second, 1.0 });
    }
    assert(imageTrackMatches.size() == tracks.size());

    return imageTrackMatches;
}

void Reconstructor::_addCameraToBundle(BundleAdjuster &ba,
    const Camera camera) {
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

tuple<double, Matx33d, ShoColumnVector3d> Reconstructor::_alignReconstructionWithHorizontalOrientation(Reconstruction & rec)
{
    cout << "Aligning reconstruction \n";
    double s;
    Mat shotOrigins(0, 3, CV_64FC1);
    Mat a;
    vector <ShoRowVector3d>gpsPositions;
    Mat gpsPositions2D, shotOrigins2D;
    Mat plane(0, 3, CV_64FC1);
    Mat verticals(0, 3, CV_64FC1);

    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        auto shotOrigin = Mat(shot.getPose().getOrigin());
        shotOrigin = shotOrigin.reshape(1, 1);
        cout << "Shot origin is " << shotOrigin << "\n\n";
        shotOrigins.push_back(shotOrigin);
        Vec2d shotOrigin2D((double*)shotOrigin.colRange(0, 2).data);
        shotOrigins2D.push_back(shotOrigin2D);
        cout << "Size of shot origins is now " << shotOrigins.size() << "\n";
        cout << "Shot origins is now " << shotOrigins << "\n";
        const auto gpsPosition = shot.getMetadata().gpsPosition;
        cout << "Gps position is now " << gpsPosition << "\n";
        gpsPositions.push_back({ gpsPosition.x, gpsPosition.y, gpsPosition.z });
        gpsPositions2D.push_back(Vec2d{ gpsPosition.x, gpsPosition.y });
        const auto[x, y, z] = shot.getOrientationVectors();

        // We always assume that the orientation type is always horizontal

        //cout << "Size of x is " << Mat(x).size() << "\n\n";
        //cout << "Type of x is " << Mat(x).type() << "\n\n";
        //cout << "Size of plane is " << plane.size() << "\n\n";
        plane.push_back(Mat(x));
        plane.push_back(Mat(z));
        verticals.push_back(-Mat(y));
    }
    Mat shotOriginsRowMean;
    reduce(shotOrigins, shotOriginsRowMean, 0, cv::REDUCE_AVG);
    cout << "Size of shotorigins row mean is " << shotOriginsRowMean.size();
    cout << "Shot origins row mean is " << shotOriginsRowMean << "\n";
    cout << "plane is " << plane << "\n";
    cout << "verticals is " << verticals << "\n";
    auto p = fitPlane(shotOriginsRowMean, plane, verticals);
    auto rPlane = calculateHorizontalPlanePosition(Mat(p));
    cout << "R plane was " << rPlane << "\n";

    Mat3d cvRPlane;
    eigen2cv(rPlane, cvRPlane);
#if 0
    cout << "Size of CV r plane was " << cvRPlane.size() << "\n";
    cout << "Size of shot origins is " << shotOrigins.size() << "\n";

    cout << "R plane was " << cvRPlane << "\n";
    cout << "Shot origins was " << shotOrigins << "\n";
#endif
    const auto shotOriginsTranspose = shotOrigins.t();
    cout << "Size of shot origins transpose is " << shotOriginsTranspose.size() << "\n";
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

        cout << "Shot origins 2d is " << shotOrigins2D << "\n";
        cout << "Gps positions is " << gpsPositions2D << "\n";
        tAffine = estimateAffinePartial2D(shotOrigins2D, gpsPositions2D);
        cout << "Type of t affine was " << tAffine.type() << "\n";
        cout << "Size of t affine was " << tAffine.size() << "\n";
        tAffine.push_back(Mat(ShoRowVector3d{ 0,0,1 }));
        cout << "T affine was " << tAffine << "\n\n";
        //TODO apply scalar operation to s
        const auto s = pow(determinant(tAffine), 0.5);
        auto a = Mat(Matx33d::eye());
        cout << "a is " << a << "\n";
        tAffine = tAffine / s;
        cv::Mat aBlock = a(cv::Rect(0, 0, 2, 2));
        tAffine.copyTo(aBlock);
        cout << "CV r plane was " << cvRPlane << "\n";
        a *= cvRPlane;
        auto b3 = mean(shotOrigins.colRange(0, 2))[0] - mean(s * Mat(gpsPositions).reshape(1).colRange(0, 2))[0];
        ShoColumnVector3d b{ tAffine.at<double>(0, 2), tAffine.at<double>(1, 2), b3 };
        return make_tuple(s, a, b);
    }
}

#if 1
void Reconstructor::_reconstructionSimilarity(Reconstruction & rec, double s, Matx33d a, ShoColumnVector3d b)
{

    for (auto &[trackId, cp] : rec.getCloudPoints()) {
        const auto pointCoordinates = Mat(convertVecToRowVector(cp.getPosition()));
        ShoColumnVector3d alignedCoordinate = (s *a) * (pointCoordinates);
        alignedCoordinate += b;
        cp.setPosition(Point3d{ alignedCoordinate(0,0), alignedCoordinate(1,0), alignedCoordinate(2,0) });
    }

    for (auto &[shotId, shot] : rec.getReconstructionShots()) {
        const auto r = shot.getPose().getRotationMatrix();
        const auto t = shot.getPose().getTranslation();
        const auto rp = r * a.t();
        const auto tp = -rp * b + s * t;
        shot.getPose().setRotationVector(Mat(rp));
        shot.getPose().setTranslation(tp);
    }
}
#endif

TwoViewPose Reconstructor::recoverTwoCameraViewPose(CommonTrack track,
    Mat &mask) {
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(track, points1, points2);
    const auto kMatrix = flight.getCamera().getNormalizedKMatrix();
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

void Reconstructor::twoViewReconstructionInliers(vector<Mat>& Rs_decomp, vector<Mat>& ts_decomp, vector<int> possibleSolutions,
    vector<Point2d> points1, vector<Point2d> points2) const
{
    
    auto bearings1 = flight.getCamera().normalizedPointsToBearingVec(points1);
    auto bearings2 = flight.getCamera().normalizedPointsToBearingVec(points2);

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

TwoViewPose Reconstructor::recoverTwoViewPoseWithHomography(CommonTrack track)
{
    const auto&[hom, points1, points2, homMask] = computePlaneHomography(track);
    cout << "Homography was " << hom << endl;
    vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions = decomposeHomographyMat(hom, flight.getCamera().getNormalizedKMatrix(), Rs_decomp, ts_decomp, normals_decomp);
    vector<int> filteredSolutions;
    cv::filterHomographyDecompByVisibleRefpoints(Rs_decomp, normals_decomp, points1, points2, filteredSolutions, homMask);
   if(filteredSolutions.size() > 0)
    //twoViewReconstructionInliers(Rs_decomp, ts_decomp, filteredSolutions, points1, points2);
       return {true, hom, Rs_decomp[filteredSolutions[0]], ts_decomp[filteredSolutions[0]]};
   
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

float Reconstructor::computeReconstructabilityScore(int tracks, Mat mask,
    int tresh) {
    auto inliers = countNonZero(mask);
    auto outliers = tracks - inliers;
    auto ratio = float(outliers) / tracks;
    return ratio;
}

void Reconstructor::computeReconstructability(
    const ShoTracker &tracker, vector<CommonTrack>& commonTracks) {
    auto imageNodes = tracker.getImageNodes();
    for (auto &track : commonTracks) {
        Mat mask;
        float score = 0;
        auto[success, essentrialMat, rotation, translation] = this->recoverTwoCameraViewPose(track, mask);
        if (success) {
            cout << "Computing reconstructability for " << track.imagePair.first << " and " << track.imagePair.second << "\n";
            score = this->computeReconstructabilityScore(track.commonTracks.size(), mask);
        }
        track.rScore = score;
    }
    sort(std::begin(commonTracks), std::end(commonTracks),
        [](CommonTrack a, CommonTrack b) { return a.rScore > b.rScore; });
}

//“Motion and Structure from Motion in a Piecewise Planar Environment. See paper
//by brown ”
std::tuple<Mat, vector<Point2f>, vector<Point2f>, Mat>
Reconstructor::computePlaneHomography(CommonTrack commonTrack) const {
    vector<Point2f> points1;
    vector<Point2f> points2;
    this->_alignMatchingPoints(commonTrack, points1, points2);
    Mat mask;
    auto hom = findHomography(points1, points2, mask, cv::RANSAC,
        REPROJECTION_ERROR_SD);
    return std::make_tuple(hom, points1, points2, mask);
}


void Reconstructor::runIncrementalReconstruction(const ShoTracker& tracker) {
    vector<Reconstruction> reconstructions;
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
            cout << "Score of this track was " << track.rScore << "\n";
            cout << "Starting reconstruction with " << track.imagePair.first
                << " and "
                << " and " << track.imagePair.second << '\n';
            auto optRec = beginReconstruction(track, tracker);
            if (optRec) {
                auto rec = *optRec;
                reconstructionImages.erase(track.imagePair.first);
                reconstructionImages.erase(track.imagePair.second);
                continueReconstruction(rec, reconstructionImages);
                string recFileName = flight.getImageDirectoryPath().parent_path().leaf().string() + "-" + 
                    to_string(reconstructions.size() + 1) + ".ply";
                rec.saveReconstruction(recFileName);
                reconstructions.push_back(rec);
            }
        }
    }
    Reconstruction allReconstruction;
    for (auto & rec : reconstructions) {
        for (const auto[shotId, shot] : rec.getReconstructionShots()) {
            if (!allReconstruction.hasShot(shotId)) {
                allReconstruction.getReconstructionShots()[shotId] = shot;
            }
        }

        for (const auto[trackId, cp] : rec.getCloudPoints()) {
            if (!allReconstruction.hasTrack(to_string(trackId))) {
                allReconstruction.addCloudPoint(cp);
            }
        }
    }
    colorReconstruction(allReconstruction);
    string recFileName = flight.getImageDirectoryPath().parent_path().leaf().string() + "all_green.ply";
    allReconstruction.saveReconstruction(recFileName);
    cout << "Total number of points in all reconstructions is " << allReconstruction.getCloudPoints().size() << "\n\n";
}

Reconstructor::OptionalReconstruction Reconstructor::beginReconstruction(CommonTrack track, const ShoTracker &tracker)
{
    Reconstruction rec(flight.getCamera());

    Mat mask;
    //TwoViewPose poseParameters = recoverTwoCameraViewPose(track, mask);
    TwoViewPose poseParameters = recoverTwoViewPoseWithHomography(track);

    auto [success, poseMatrix, rotation, translation] = poseParameters;
    Mat essentialMat = std::get<1>(poseParameters);
    Mat r = std::get<2>(poseParameters);
    Mat t = std::get<3>(poseParameters);

    if (poseMatrix.rows != 3 || !success)
    {
        cout << "Could not compute the essential matrix for this pair" << endl;
        //Get the first essential Mat;
        return std::nullopt;
    }

# if 0
    auto inliers = countNonZero(mask);
    if (inliers <= 5)
    {
        cout << "This pair failed to adequately reconstruct" << endl;
        return std::nullopt;
    }
#endif
    Mat rVec;
    Rodrigues(r, rVec);
    Mat distortion;

    cout << "Rvec was " << rVec + '\n';
    cout << "T was " << t << '\n';
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

    colorReconstruction(rec);
    rec.saveReconstruction("green.ply");
    cout << "Generated " << rec.getCloudPoints().size()
        << "points from initial motion " << endl;


#if 1
    singleViewBundleAdjustment(track.imagePair.second, rec);
    retriangulate(rec);
    singleViewBundleAdjustment(track.imagePair.second, rec);
#endif
    return rec;
}

void Reconstructor::continueReconstruction(Reconstruction& rec, set<string>& images) {
    bundle(rec);
    //removeOutliers(rec);
    //alignReconstruction(rec);
    colorReconstruction(rec);
    rec.saveReconstruction("partialgreen.ply");


    while (1) {
        auto candidates = reconstructedPointForImages(rec, images);
        cout << "size of candidates is " << candidates.size() << "\n";
        if (candidates.empty())
            break;

        for (auto[imageName, numTracks] : candidates) {
            auto imageVertex = getImageNode(imageName);
            auto [status, report] = resect(rec, imageVertex);
            if (!status)
                continue;

            singleViewBundleAdjustment(imageName, rec);

            images.erase(imageName);
            triangulateShots(imageName, rec);
            cout << "Rec now has " << rec.getCloudPoints().size() << "points \n";
        }
        return;
    }
}

void Reconstructor::triangulateShots(string image1, Reconstruction &rec) {
    cout << "Triangulating shots " << endl;
    auto im1 = this->imageNodes[image1];

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto track = this->tg[*tracksIter].trackName;
        this->triangulateTrack(track, rec);
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
        if (rec.hasShot(shotId)) {
            auto shot = rec.getReconstructionShots()[shotId];
            auto edgePair = boost::edge(track, this->imageNodes[shotId], this->tg);
            auto edgeDescriptor = edgePair.first;
            auto fCol = this->tg[edgeDescriptor].fProp.color;
            auto fPoint = this->tg[edgeDescriptor].fProp.coordinates;
            auto fBearing =
                this->flight.getCamera().normalizedPointToBearingVec(fPoint);
            auto origin = this->getShotOrigin(shot);
            // cout << "Origin for this shot was " << origin << endl;
            Eigen::Vector3d eOrigin;
            Eigen::Matrix3d eigenRotationInverse;
            cv2eigen(Mat(origin), eOrigin);
            auto rotationInverse = this->getRotationInverse(shot);
            cv2eigen(rotationInverse, eigenRotationInverse);
            //cout << "Rotation inverse is " << eigenRotationInverse << endl;
            auto eigenRotationBearingProduct = eigenRotationInverse * fBearing;
            // cout << "Rotation inverse times bearing us  " <<
            // eigenRotationBearingProduct << endl;
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
            cout << "added cloud point, size now  " << rec.getCloudPoints().size() << "\n";
        }
    }
}

void Reconstructor::retriangulate(Reconstruction& rec) {
    set<string> tracks;
    for (const auto[imageName, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes.at(imageName);
            const auto[edgesBegin, edgesEnd] =
                boost::out_edges(imageVertex, this->tg);

            for (auto edgesIter = edgesBegin; edgesIter != edgesEnd; ++edgesIter) {
                const auto trackName = this->tg[*edgesIter].trackName;
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

    const auto[edgesBegin, edgesEnd] = boost::out_edges(im1, this->tg);

    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        auto trackId = this->tg[*tracksIter].trackName;
        try {
            const auto track = rec.getCloudPoints().at(stoi(trackId));
            const auto p = track.getPosition();
            const auto featureCoords = this->tg[*tracksIter].fProp.coordinates;
            bundleAdjuster.AddPoint(trackId, p.x, p.y, p.z, true);
            bundleAdjuster.AddObservation(shotId, trackId, featureCoords.x,
                featureCoords.y);
        }
        catch (std::out_of_range &e) {
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
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    const auto frameName =
        track.imagePair.first + " - " + track.imagePair.second + " tracks";
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
}

void Reconstructor::bundle(Reconstruction& rec) {
    BundleAdjuster bundleAdjuster;

    _addCameraToBundle(bundleAdjuster, rec.getCamera());

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        const auto r = shot.getPose().getRotationVector();
        const auto t = shot.getPose().getTranslation();


        bundleAdjuster.AddShot(shot.getId(), "1", r(0), r(1),
            r(2), t(0), t(1), t(2), false);
    }

    for (const auto[id, cloudPoint] : rec.getCloudPoints()) {
        const auto coord = cloudPoint.getPosition();
        bundleAdjuster.AddPoint(std::to_string(id), coord.x, coord.y, coord.z,
            false);
    }

    for (const auto[shotId, shot] : rec.getReconstructionShots()) {
        try {
            const auto imageVertex = imageNodes.at(shotId);
            const auto[edgesBegin, edgesEnd] =
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
        }
        catch (std::out_of_range &e) {
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

    for (auto &[shotId, shot] : rec.getReconstructionShots()) {
        auto s = bundleAdjuster.GetShot(shotId);
        Mat rotation = (Mat_<double>(3, 1) << s.GetRX(), s.GetRY(), s.GetRZ());
        cout << "Rotation was " << rotation << "\n";
        Mat translation = (Mat_<double>(3, 1) << s.GetTX(), s.GetTY(), s.GetTZ());

        shot.getPose().setRotationVector(rotation);
        shot.getPose().setTranslation(translation);

        cout << "Origin in map is " << rec.getReconstructionShots()[shotId].getPose().getOrigin() << "\n";

        Pose testPose{ translation, rotation };
        cout << "Origin from this pose is " << testPose.getOrigin() << "\n";
    }

    for (auto &[pointId, cloudPoint] : rec.getCloudPoints()) {
        auto point = bundleAdjuster.GetPoint(std::to_string(pointId));
        cloudPoint.getPosition().x = point.GetX();
        cloudPoint.getPosition().y = point.GetY();
        cloudPoint.getPosition().z = point.GetZ();
        cloudPoint.setError(point.reprojection_error);
    }
}

vector<pair<string, int>> Reconstructor::reconstructedPointForImages(const Reconstruction & rec, set<string> & images)
{
    vector<pair <string, int>> res;
    for (const auto imageName : images) {
        if (!rec.hasShot(imageName)) {
            auto commonTracks = 0;
            const auto[edgesBegin, edgesEnd] = boost::out_edges(getImageNode(imageName), tg);
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
void  Reconstructor::alignReconstruction(Reconstruction & rec)
{
    const auto[s, a, b] = _alignReconstructionWithHorizontalOrientation(rec);
#if 0
    cout << "s is " << s << "\n";
    cout << "a is " << a << "\n";
    cout << "b ia " << b << "\n";
#endif
    _reconstructionSimilarity(rec, s, a, b);

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

void Reconstructor::colorReconstruction(Reconstruction & rec)
{
    for (auto&[trackId, cp] : rec.getCloudPoints()) {
        const auto trackNode = this->getTrackNode(to_string(trackId));
        auto[edgesBegin, _] = boost::out_edges(trackNode, this->tg);
        edgesBegin++;
        cp.setColor(tg[*edgesBegin].fProp.color);
    }
}

bool Reconstructor::shouldTriangulate()
{
    return false;
}

void Reconstructor::removeOutliers(Reconstruction& rec) {
    const auto before = rec.getCloudPoints().size();
    erase_if(
        rec.getCloudPoints(),
        [](const auto& cp) {
        const auto&[id, p] = cp;
        return p.getError() > BUNDLE_OUTLIER_THRESHOLD;
    }
    );
    const auto removed = before - rec.getCloudPoints().size();
    cout << "Removed " << removed << " outliers from reconstruction \n";
}

tuple<bool, ReconstructionReport> Reconstructor::resect(Reconstruction & rec, const vertex_descriptor imageVertex, double threshold,
    int iterations, double probability, int resectionInliers) {
    ReconstructionReport report;
    opengv::points_t Xs;
    opengv::bearingVectors_t Bs;
    vector<Point2d> fPoints;
    vector<Point3d> realWorldPoints;
    const auto[edgesBegin, edgesEnd] = boost::out_edges(imageVertex, this->tg);
    cout << "Reconstruction has " << rec.getCloudPoints().size() << "tracks \n";
    for (auto tracksIter = edgesBegin; tracksIter != edgesEnd; ++tracksIter) {
        const auto trackName = this->tg[*tracksIter].trackName;
        if (rec.hasTrack(trackName)) {
            auto fPoint = this->tg[*tracksIter].fProp.coordinates;
            auto fBearing =
                this->flight.getCamera().normalizedPointToBearingVec(fPoint);
            //cout << "F point to f bearing is " << fPoint << " to " << fBearing << "\n";

            fPoints.push_back(flight.getCamera().denormalizeImageCoordinates(fPoint));
            auto position = rec.getCloudPoints().at(stoi(trackName)).getPosition();
            realWorldPoints.push_back(position);
            Xs.push_back({ position.x, position.y, position.z });
            Bs.push_back(fBearing);
        }
    }
    cout << "Size of bs is " << Bs.size() << "\n";
    if (Bs.size() < 5) {
        report.numCommonPoints = Bs.size();
        return make_tuple(false, report);
    }

    Mat pnpRot, pnpTrans, inliers;
    if (cv::solvePnPRansac(realWorldPoints, fPoints, flight.getCamera().getKMatrix(), 
        flight.getCamera().getDistortionMatrix(), pnpRot, pnpTrans, false,iterations, 8.0, probability, inliers)) {

        const auto shotName = tg[imageVertex].name;
        const auto shot = flight.getImageSet()[flight.getImageIndex(shotName)];
        ShotMetadata shotMetadata(shot.getMetadata(), flight);
        cout << "Rotation from absolute ransac is " << pnpRot << "\n";
        cout << "Translation from absolute ransac is " << pnpTrans << "\n";
        report.numCommonPoints = Bs.size();
        report.numInliers = cv::countNonZero(inliers);
        Shot recShot(shotName, flight.getCamera(), Pose(pnpRot, pnpTrans), shotMetadata);
        rec.getReconstructionShots()[recShot.getId()] = recShot;
        return { true, report };
    }
    const auto t = absolutePoseRansac(Bs, Xs, threshold, iterations, probability);
    cout << "T obtained was " << t << "\n";
    Matrix3d rotation;
    RowVector3d translation;

    rotation = t.leftCols(3);
    translation = t.rightCols(1).transpose();
    auto fd = Xs.data()->data();
    auto bd = Bs.data()->data();
    Matrix<double, Dynamic, 3> eigenXs = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(fd, Xs.size(), 3);
    Matrix<double, Dynamic, 3> eigenBs = Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bd, Bs.size(), 3);

    cout << "Eigen xs is " << eigenXs << "\n";
    cout << "Eigen bs is " << eigenBs << "\n";

    const auto rotationTranspose = rotation.transpose();
    cout << "Rotation transpose is " << rotationTranspose << "\n";
    const auto eigenXsMinusTranslationTranspose = (eigenXs.rowwise() - translation).transpose();
    cout << "Eigen minus translation is " << eigenXsMinusTranslationTranspose << "\n";
    const auto rtProduct = rotation.transpose() * eigenXsMinusTranslationTranspose;
    cout << "rt product is " << rtProduct << "\n";
    MatrixXd reprojectedBs(rtProduct.cols(), rtProduct.rows());
    reprojectedBs = rtProduct.transpose();
    cout << "Reprojected bs is " << reprojectedBs << "\n";
    VectorXd reprojectedBsNorm(reprojectedBs.rows());
    reprojectedBsNorm = reprojectedBs.rowwise().norm();
    cout << "Reprojected bs norm is " << reprojectedBsNorm << "\n";
    auto divReprojectedBs = reprojectedBs.array().colwise() / reprojectedBsNorm.array();

    cout << "Div reprojected bs is " << divReprojectedBs << "\n";
    MatrixXd reprojectedDifference(eigenBs.rows(), eigenBs.cols());
    reprojectedDifference = reprojectedBs - eigenBs;
    cout << "Reprojected difference is " << reprojectedDifference << "\n";
    cout << "Reprojected difference norm is" << reprojectedDifference.rowwise().norm()<< "\n";
    /*

    const auto inliersMatrix = divReprojectedBs.rowwise() - eigenBs.colwise().norm();
        inliersMatrix < threshold).count();
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

    return make_tuple(false, report);
}