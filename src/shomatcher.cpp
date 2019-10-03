#include "shomatcher.hpp"
#include "kdtree.h"
#include "camera.h"
#include "utilities.h"
#include "RobustMatcher.h"
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include "json.hpp"
#include "bootstrap.h"
#include <set>

using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::Mat;
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
using std::cerr;
using std::set;
using std::max;
using std::vector;
using std::string;
using json = nlohmann::json;

ShoMatcher::ShoMatcher(FlightSession flight, int featureSize, RobustMatcher::Feature featureType)
    : flight_(flight)
    , featureSize_(featureSize)
    , kd_(nullptr)
    , candidateImages()
    , rMatcher_(RobustMatcher::create(featureType)){}

void ShoMatcher::getCandidateMatchesUsingSpatialSearch(double range)
{
    set<pair<string, string>> alreadyPaired;
    buildKdTree();
    auto imageSet = flight_.getImageSet();
    for (const auto img : imageSet) {
        vector<string> matchSet;
        auto currentImageName = img.getFileName();
        void *result_set;

        double pt[] = { img.getMetadata().location.longitude, img.getMetadata().location.latitude };
        result_set = kd_nearest_range(static_cast<kdtree *>(kd_), pt, range);
        vector<double> pos(this->dimensions_);
        int count = 0;
        while (!kd_res_end(static_cast<kdres *>(result_set)))
        {
            auto current = kd_res_item(static_cast<kdres *>(result_set), pos.data());
            if (current == nullptr)
                continue;
            auto img = static_cast<Img *>(current);
            if (currentImageName != img->getFileName())
            {
                count++;
                //Make sure we are matching pairs already matched in reverse order
                if (alreadyPaired.find(make_pair(currentImageName, img->getFileName())) == alreadyPaired.end()
                    || alreadyPaired.find(make_pair(img->getFileName(), currentImageName)) == alreadyPaired.end()) {
                    alreadyPaired.insert(make_pair(currentImageName, img->getFileName()));
                    matchSet.push_back(img->getFileName());
                }

            }
            kd_res_next(static_cast<kdres *>(result_set));
        }
        cout << "Found " << count << " candidate matches for " << currentImageName << endl;
        if (matchSet.size())
        {
            this->candidateImages[currentImageName] = matchSet;
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
    //set feature process size to -1 to avoid resizing
    if (featureSize_ != -1) {
        auto maxSize = max(flight_.getCamera().getHeight(), flight_.getCamera().getWidth());
        int fx = flight_.getCamera().getWidth() * featureSize_ / maxSize;
        int fy = flight_.getCamera().getHeight() * featureSize_ / maxSize;
        flight_.getCamera().setScaledHeight(fy);
        flight_.getCamera().setScaledWidth(fx);
    }

    set<string> detected;
    if (!candidateImages.size())
        return 0;
    const auto cudaEnabled = checkIfCudaEnabled();
    //Only use parallelism if not on a cuda device. Working with one cuda device from  multiple threads is undefined behaviour
//#pragma omp parallel for if(!cudaEnabled) shared(detected)
#pragma omp parallel for shared(detected)
        for (auto i = 0; i < candidateImages.size(); ++i)
            //for (auto it = candidateImages.begin(); it != candidateImages.end(); it++)
        {
            auto it = candidateImages.begin();
            advance(it, i);
#pragma omp critical 
            {
                if (detected.find(it->first) == detected.end() && _extractFeature(it->first))
                {
                    detected.insert(it->first);
                }
            }


            for (auto j = 0; j < it->second.size(); ++j)
                //for (auto _it = it->second.begin(); _it != it->second.end(); ++_it)
            {
                auto _it = it->second.begin();
                advance(_it, j);
#pragma omp critical
                {
                    if (detected.find(*_it) == detected.end())
                    {
                        if (_extractFeature(*_it))
                        {
                            detected.insert(*_it);
                        }
                    }
                }

            }
    }
    return detected.size();
}

bool ShoMatcher::_extractFeature(string fileName)
{
    auto imageFeaturePath = flight_.getImageFeaturesPath() / (fileName + ".yaml");
    auto modelimageNamePath = flight_.getImageDirectoryPath() / (fileName);
    if (boost::filesystem::exists(imageFeaturePath)) {
        //Use existing file instead.
        cerr << "Using " << imageFeaturePath.string() << " for features \n";
        return true;
    }

    Mat modelImg = imread(modelimageNamePath.string(), SHO_LOAD_COLOR_IMAGE_OPENCV_ENUM | SHO_LOAD_ANYDEPTH_IMAGE_OPENCV_ENUM);
    cv::cvtColor(modelImg, modelImg, SHO_BGR2RGB);
    Mat featureImage = imread(modelimageNamePath.string(), SHO_GRAYSCALE);

    auto channels = modelImg.channels();

    if (modelImg.empty())
        return false;

    if (featureSize_ != -1) {
        cv::resize(modelImg, modelImg, { flight_.getCamera().getScaledWidth(), flight_.getCamera().getScaledHeight() }, 0, 0, cv::INTER_AREA);
        cv::resize(featureImage, featureImage, { flight_.getCamera().getScaledWidth(), flight_.getCamera().getScaledHeight() }, 0, 0, cv::INTER_AREA);
        cout << "Size of feature image is " << featureImage.size() << "\n";
    }

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Scalar> colors;
    cv::Mat descriptors;

    rMatcher_->detectAndCompute(featureImage, keypoints, descriptors);

    cout << "Extracted " << descriptors.rows << " points for  " << fileName << endl;
    for (auto &keypoint : keypoints) {
        if (channels == 1)
            colors.push_back(modelImg.at<uchar>(keypoint.pt));
        else if (channels == 3)
            colors.push_back(modelImg.at<Vec3b>(keypoint.pt));

        keypoint.pt = this->flight_.getCamera().normalizeImageCoordinate(keypoint.pt);
    }
    return flight_.saveImageFeaturesFile(fileName, keypoints, descriptors, colors);
}


void ShoMatcher::runRobustFeatureMatching()
{
    if (!candidateImages.size())
        return;

    map<string, ImageFeatures> loadedFeatures;
    for (const auto&[queryImg, trainImages] : candidateImages) {
        vector<string> trainImageSet;
        auto queryImagePath = flight_.getImageDirectoryPath() / queryImg;
        ImageFeatures queryFeaturesSet;
        try {
            queryFeaturesSet = loadedFeatures.at(queryImg);
        }
        catch (std::out_of_range e) {
            queryFeaturesSet = flight_.loadFeatures(queryImg);
            loadedFeatures[queryImg] = queryFeaturesSet;
        }
        map<string, vector<DMatch>> matchSet;
        for (const auto trainImg : trainImages)
        {
            trainImageSet.push_back(trainImg);
            ImageFeatures trainFeaturesSet;
            try {
                trainFeaturesSet = loadedFeatures.at(trainImg);
            }
            catch (std::out_of_range e) {
                trainFeaturesSet = flight_.loadFeatures(trainImg);
                loadedFeatures[trainImg] = trainFeaturesSet;
            }
            vector<DMatch> matches;
            rMatcher_->robustMatch(queryFeaturesSet.descriptors, trainFeaturesSet.descriptors, matches);
            int trainIndex = this->flight_.getImageIndex(trainImg);
            for (size_t i = 0; i < matches.size(); i++)
            {
                //Update train index so we know what image we matched against when we are running the tracking pipeline
                matches[i].imgIdx = trainIndex;
            }
            matchSet[trainImg] = matches;
            cout << queryImg << " - " << trainImg << " has " << matches.size() << "candidate matches" << endl;
        }
        this->flight_.saveMatches(queryImg, matchSet);
    }
}

void ShoMatcher::buildKdTree()
{
    kd_ = kd_create(dimensions_);
    for (auto &img : flight_.getImageSet())
    {
        auto pos = vector<double>{ img.getMetadata().location.longitude, img.getMetadata().location.latitude };
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree *>(kd_), pos.data(), dt) == 0);
    }

}

map<string, std::vector<string>> ShoMatcher::getCandidateImages() const
{
    return this->candidateImages;
}

void ShoMatcher::plotMatches(string img1, string img2) const {
    Mat imageMatches;
    Mat image1 = imread((flight_.getImageDirectoryPath() / img1).string(),
        cv::IMREAD_GRAYSCALE);
    Mat image2 = imread((flight_.getImageDirectoryPath() / img2).string(),
        cv::IMREAD_GRAYSCALE);
    auto img1Matches = flight_.loadMatches(img1);
    auto kp1 = flight_.loadFeatures(img1).getKeypoints();
    auto kp2 = flight_.loadFeatures(img2).getKeypoints();
    for (auto& kp : kp1) {
        kp.pt = flight_.getCamera().denormalizeImageCoordinates(kp.pt);
    }

    for (auto& kp : kp2) {
        kp.pt = flight_.getCamera().denormalizeImageCoordinates(kp.pt);
    }
    if (img1Matches.find(img2) != img1Matches.end()) {
        auto matches = img1Matches[img2];
        drawMatches(image1, kp1, image2, kp2, matches, imageMatches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    const auto frameName = img1 + " - " + img2 + " matches";
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
}

