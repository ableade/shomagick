#include "shomatcher.hpp"
#include "kdtree.h"
#include "RobustMatcher.h"
#include <set>

using cv::Ptr;
using cv::FeatureDetector;
using cv::imread;
using cv::Mat;
using cv::DMatch;
using cv::ORB;
using std::map;
using std::pair;
using std::set;
using std::endl;
using std::cout;
using std::vector;

void ShoMatcher::getCandidateMatches(double range)
{   
    this->buildKdTree();
    auto imageSet = this->flight.getImageSet();
    for (int i = 0; i < imageSet.size(); ++i)
    {
        vector<string> matchSet;
        auto currentImage = imageSet[i].fileName;
        void *result_set;

        double pt[] = {imageSet[i].location.longitude, imageSet[i].location.latitude};
        result_set = kd_nearest_range(static_cast<kdtree *>(kd), pt, range);
        auto resultSetSize = static_cast<kdres*>(result_set)->size;
        double pos[this->dimensions];
        while (!kd_res_end(static_cast<kdres *>(result_set)))
        {
            auto count = 0;
            auto current = kd_res_item(static_cast<kdres *>(result_set), pos);
            if (current == nullptr)
                continue;
            
            auto img = static_cast<Img *>(current);
            double dist = sqrt(dist_sq(pt, pos, this->dimensions));
            if (currentImage != img->fileName) {
                matchSet.push_back(img->fileName);
            }
            kd_res_next(static_cast<kdres *>(result_set));
        }
        cout << "Found " << matchSet.size()<< " candidate matches for " << currentImage << endl; 
        if (matchSet.size()) {
            this->candidateImages[currentImage] = matchSet;
        }
    }
}

int ShoMatcher::extractFeatures() {
    set <string> detected;
    if(! this->candidateImages.size()) 
        return 0;
    
    for(auto it = this->candidateImages.begin(); it!=candidateImages.end(); it++) {
        if (detected.find(it->first) == detected.end() && this->_extractFeature(it->first)) {
            detected.insert(it->first);
        }
        
        for (auto _it = it->second.begin(); _it!=it->second.end(); ++_it) {
            if (detected.find(*_it) == detected.end()) {
                if (this->_extractFeature(*_it)) {
                    detected.insert(*_it);
                }
            }
        }
        
    }
    return detected.size();
}

bool ShoMatcher:: _extractFeature(string fileName) {
    auto modelimageNamePath = this->flight.getImageDirectoryPath() / fileName;
    Mat modelImg = imread(modelimageNamePath.string());
    if (modelImg.empty())
        return false;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    this->detector_->detect(modelImg, keypoints);
    this->extractor_->compute(modelImg, keypoints, descriptors);
    cout << "Extracted features for "<<fileName <<endl;
    return this->flight.saveImageFeaturesFile(fileName, keypoints, descriptors);
}

void ShoMatcher::runRobustFeatureDetection() {
    if (!this->candidateImages.size())
        return;
    
    RobustMatcher rmatcher;

    for(auto it = this->candidateImages.begin(); it!=candidateImages.end(); it++) {
        auto modelimageNamePath = this->flight.getImageDirectoryPath() / it->first;
        auto trainFeatureSet = this->flight.loadFeatures(it->first);
        map<string , vector<DMatch>> matchSet;
        for (auto matchIt = it->second.begin(); matchIt!= it->second.end(); ++ matchIt) {
            auto queryFeatureSet = this->flight.loadFeatures(*matchIt);
            vector<DMatch> matches;

            rmatcher.robustMatch(trainFeatureSet.first, trainFeatureSet.second, queryFeatureSet.first, queryFeatureSet.second, matches);

            int trainIndex = this->flight.getImageIndex(*matchIt);
            for(int i=0; i<matches.size(); i++) {
                //Update train index so we know what image we matched against when we are running the tracking pipeline
                matches[i].imgIdx = trainIndex;
                
            }
            matchSet[*matchIt] = matches;
            this->flight.saveMatches(it->first, matchSet);
        }
    }
    
}

void ShoMatcher::buildKdTree() {
    kd= kd_create(this->dimensions);
    for (auto& img : this->flight.getImageSet()) {
        double pos[this->dimensions] = {img.location.longitude, img.location.latitude};
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree*>(kd), pos, dt) == 0); 
    }
}

map<string, std::vector<string>> ShoMatcher::getCandidateImages() const {
   return this->candidateImages;
}

void ShoMatcher::setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detector) {
   this->detector_ = detector;
}

void ShoMatcher::setFeatureExtractor(const cv::Ptr<cv::DescriptorExtractor>& extractor) {
   this->extractor_ = extractor;
}

void ShoMatcher::setMatcher(const cv::Ptr<cv::DescriptorMatcher>& matcher) {
   this->matcher_ = matcher;
}