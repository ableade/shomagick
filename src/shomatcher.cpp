#include "shomatcher.hpp"
#include "kdtree.h"
#include "RobustMatcher.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::Mat;
using cv::ORB;
using cv::Ptr;
using cv::Vec3b;
using std::cout;
using std::endl;
using std::map;
using std::pair;
using std::set;
using std::vector;

void ShoMatcher::getCandidateMatches(double range)
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

   // auto modelimageNamePath = this->flight.getImageDirectoryPath() / fileName;
    auto modelimageNamePath =  fileName;
    Mat modelImg = imread(modelimageNamePath, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR );

    auto channels = modelImg.channels();

    if (modelImg.empty())
        return false;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Scalar> colors;
    cv::Mat descriptors;
    this->detector_->detect(modelImg, keypoints);
    this->extractor_->compute(modelImg, keypoints, descriptors);
    cout << "Extracted "<< descriptors.rows << " points for  " << fileName<< endl;
    for(auto const &keypoint : keypoints) {
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

<<<<<<< HEAD
void ShoMatcher::setMatcher(const cv::Ptr<cv::DescriptorMatcher> &matcher)
{
    this->matcher_ = matcher;
}
=======
void ShoMatcher::setFeatureDetector(const cv::Ptr<cv::FeatureDetector> &detector)
{
    this->detector_ = detector;
}

void ShoMatcher::setFeatureExtractor(const cv::Ptr<cv::DescriptorExtractor> &extractor)
{
    this->extractor_ = extractor;
}
>>>>>>> 4b284a0... implement cross platform path check
