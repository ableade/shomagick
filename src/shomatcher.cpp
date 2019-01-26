#include "shomatcher.hpp"
#include "kdtree.h"
#include "camera.h"
#include "RobustMatcher.h"
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "json.hpp"
#include <set>

using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::Mat;
using cv::ORB;
using cv::Ptr;
using cv::Vec3b;
using cv::Scalar;
using cv::drawMatches;
using cv::imshow;
using cv::waitKey;
using cv::DrawMatchesFlags;
using std::cout;
using std::endl;
using std::map;
using std::pair;
using std::set;
using std::vector;
using json = nlohmann::json;

void ShoMatcher::getCandidateMatchesUsingSpatialSearch(double range)
{
    this->buildKdTree();
    auto imageSet = this->flight.getImageSet();
    for (size_t i = 0; i < imageSet.size(); ++i)
    {
        vector<string> matchSet;
        auto currentImage = imageSet[i].fileName;
        void *result_set;

        double pt[] = {imageSet[i].location.longitude, imageSet[i].location.latitude};
        result_set = kd_nearest_range(static_cast<kdtree *>(kd), pt, range);
        vector<double> pos(this->dimensions);
        while (!kd_res_end(static_cast<kdres *>(result_set)))
        {
            auto current = kd_res_item(static_cast<kdres *>(result_set), pos.data());
            if (current == nullptr)
                continue;

            auto img = static_cast<Img *>(current);
            if (currentImage != img->fileName)
            {
                matchSet.push_back(img->fileName);
            }
            kd_res_next(static_cast<kdres *>(result_set));
        }
        cout << "Found " << matchSet.size() << " candidate matches for " << currentImage << endl;
        if (matchSet.size())
        {
            this->candidateImages[currentImage] = matchSet;
        }
    }
}

void ShoMatcher::getCandidateMatchesFromFile(string candidatesFile) {
    assert(boost::filesystem::exists(candidatesFile));
    std::ifstream infile(candidatesFile);
    json matchesReport;
    infile >> matchesReport;
    const auto pairs = matchesReport["pairs"];
    for (auto aPair : pairs) {
        this->candidateImages[aPair[0]].push_back(aPair[1]);
    }
}

int ShoMatcher::extractFeatures()
{
    set<string> detected;
    if (!this->candidateImages.size())
        return 0;

    for (auto it = this->candidateImages.begin(); it != candidateImages.end(); it++)
    {
        if (detected.find(it->first) == detected.end() && this->_extractFeature(it->first))
        {
            detected.insert(it->first);
        }

        for (auto _it = it->second.begin(); _it != it->second.end(); ++_it)
        {
            if (detected.find(*_it) == detected.end())
            {
                if (this->_extractFeature(*_it))
                {
                    detected.insert(*_it);
                }
            }
        }
    }
    return detected.size();
}

bool ShoMatcher::_extractFeature(string fileName)
{
    auto modelimageNamePath = this->flight.getImageDirectoryPath() / fileName;
    Mat modelImg = imread(modelimageNamePath.string(), SHO_LOAD_COLOR_IMAGE_OPENCV_ENUM | SHO_LOAD_ANYDEPTH_IMAGE_OPENCV_ENUM);
    cv::cvtColor(modelImg, modelImg, SHO_BGR2RGB);
    auto channels = modelImg.channels();

    if (modelImg.empty())
        return false;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Scalar> colors;
    cv::Mat descriptors;
    this->detector_->detect(modelImg, keypoints);
    this->extractor_->compute(modelImg, keypoints, descriptors);
    cout << "Extracted "<< descriptors.rows << " points for  " << fileName<< endl;

    for(auto  &keypoint : keypoints) {
		keypoint.pt = this->flight.getCamera().normalizeImageCoordinate(keypoint.pt);
        if (channels == 1)
            colors.push_back(modelImg.at<uchar>(keypoint.pt));
        else if (channels == 3) 
            colors.push_back(modelImg.at<Vec3b>(keypoint.pt));
    }   
    return this->flight.saveImageFeaturesFile(fileName, keypoints, descriptors, colors);
}

void ShoMatcher::runRobustFeatureMatching()
{
    if (!this->candidateImages.size())
        return;

    RobustMatcher rmatcher;

    for (auto it = this->candidateImages.begin(); it != candidateImages.end(); it++)
    {
        auto modelimageNamePath = this->flight.getImageDirectoryPath() / it->first;
        auto trainFeatureSet = this->flight.loadFeatures(it->first);
        map<string, vector<DMatch>> matchSet;
        for (auto matchIt = it->second.begin(); matchIt != it->second.end(); ++matchIt)
        {
            auto queryFeatureSet = this->flight.loadFeatures(*matchIt);
            vector<DMatch> matches;

            rmatcher.robustMatch(trainFeatureSet.keypoints, trainFeatureSet.descriptors, queryFeatureSet.keypoints, queryFeatureSet.descriptors, matches);
            
            int trainIndex = this->flight.getImageIndex(*matchIt);
            for (size_t i = 0; i < matches.size(); i++)
            {
                //Update train index so we know what image we matched against when we are running the tracking pipeline
                matches[i].imgIdx = trainIndex;
            }
            matchSet[*matchIt] = matches;
            cout << it->first << " - " << *matchIt << " has " << matches.size() << "candidate matches"<<endl;
        }
        this->flight.saveMatches(it->first, matchSet);
    }
}

void ShoMatcher::buildKdTree()
{
    kd = kd_create(this->dimensions);
    for (auto &img : this->flight.getImageSet())
    {
        auto pos = vector<double> {img.location.longitude, img.location.latitude};
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree *>(kd), pos.data(), dt) == 0);
    }
}

map<string, std::vector<string>> ShoMatcher::getCandidateImages() const
{
    return this->candidateImages;
}

void ShoMatcher::plotMatches(string img1, string img2) const{
    Mat imageMatches;
    Mat image1 = imread((this->flight.getImageDirectoryPath() / img1).string(),
        cv::IMREAD_GRAYSCALE);
    Mat image2 = imread((this->flight.getImageDirectoryPath() / img2).string(), 
        cv::IMREAD_GRAYSCALE);
    auto img1Matches = this->flight.loadMatches(img1);
    auto kp1 = this->flight.loadFeatures(img1).getKeypoints();
    auto kp2 = this->flight.loadFeatures(img2).getKeypoints();
    for (auto& kp : kp1) {
        kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    for (auto& kp : kp2) {
        kp.pt = this->flight.getCamera().denormalizeImageCoordinates(kp.pt);
    }
    if (img1Matches.find(img2) != img1Matches.end()) {
        auto matches = img1Matches[img2];
        drawMatches(image1, kp1, image2, kp2, matches, imageMatches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }
    
    const auto frameName = img1 + " - " + img2;
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
    waitKey(0);
}

