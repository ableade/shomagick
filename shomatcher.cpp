#include "shomatcher.hpp"
#include "kdtree.h"
#include "RobustMatcher.h"

using cv::Ptr;
using cv::FeatureDetector;
using cv::imread;
using cv::ORB;
using std::map;
using std::pair;
using std::endl;

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
        cout << "Size of result set is "<<resultSetSize<<endl;
        double pos[this->dimensions];
        while (!kd_res_end(static_cast<kdres *>(result_set)))
        {
            auto count = 0;
            auto current = kd_res_item(static_cast<kdres *>(result_set), pos);
            if (current == nullptr)
                continue;
            
            auto img = static_cast<Img *>(current);
            cout << "Image filename is "<<img->fileName.size()<<endl;
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
void ShoMatcher::runRobustFeatureDetection() {
    RobustMatcher rmatcher;
    cv::Ptr<FeatureDetector> orb = ORB::create();
    rmatcher.setFeatureDetector(orb); 
    rmatcher.setDescriptorExtractor(orb); 

    for(auto it = this->candidateImages.begin(); it!=candidateImages.end(); it++) {
        auto modelimageNamePath = this->flight.getImageDirectoryPath() / it->first;
        cout << "Reading image "<<modelimageNamePath.string()<<endl;
        Mat modelImg = imread(modelimageNamePath.string());
        if (modelImg.empty())
            cout << "The model image is empty"<<endl;
        for (auto matchIt = it->second.begin(); matchIt!= it->second.end(); ++ matchIt) {
            auto queryImageNamePath = this->flight.getImageDirectoryPath() / *matchIt;
            Mat queryImg = imread(queryImageNamePath.string());
            if (queryImg.empty()) {
                cout << "The query image is empty "<<endl;
            }
            vector<DMatch> matches;
            vector<KeyPoint> keypoints1, keypoints2;
            rmatcher.robustMatch(queryImg, modelImg, matches, keypoints1, keypoints2);

            cout << "Size of match set obtained from "<< it->first << " is "<< matches.size()<<endl;
            int trainIndex = this->flight.getImageIndex(*matchIt);
            for(int i=0; i<matches.size(); i++) {
                //Update train index so we know what image we matched against when we are running the tracking pipeline
                matches[i].imgIdx = trainIndex;
                
            }
            this->saveMatches(it->first, matches);
        }
    }
    
}

bool ShoMatcher::generateImageFeaturesFile(string imageName) {
    vector<KeyPoint> keypoints;
    Mat descriptors;
    auto imageNamePath = this->flight.getImageDirectoryPath() / imageName;
    auto imageFeaturePath = this->flight.getImageFeaturesPath() / imageName;
    if (!boost::filesystem::exists(imageFeaturePath))
    {
        Mat img = imread(imageNamePath.string());
        if (!img.empty())
        {
            surfDetector(img, Mat(), keypoints, descriptors);
            cv::FileStorage file(imageFeaturePath.string(), cv::FileStorage::WRITE);
            file << imageFeaturePath.string() << descriptors;
            file.release();
            }
        }
    return boost::filesystem::exists(imageFeaturePath);
}

void ShoMatcher::buildKdTree() {
    kd= kd_create(this->dimensions);
    for (auto& img : this->flight.getImageSet()) {
        double pos[this->dimensions] = {img.location.longitude, img.location.latitude};
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree*>(kd), pos, dt) == 0); 
    }
}

bool ShoMatcher::saveMatches(string fileName, std::vector<cv::DMatch> matches) {

    auto imageMatchesPath = this->flight.getImageMatchesPath() / (fileName + ".xml");
    cout << "Saving file "<< imageMatchesPath.string()<<endl;;
    cv::FileStorage fs(imageMatchesPath.string(), cv::FileStorage::WRITE);
    fs<< "matchCount" << (int)matches.size();
    fs << "matches" << matches;
    fs.release();
    return boost::filesystem::exists(imageMatchesPath);
}