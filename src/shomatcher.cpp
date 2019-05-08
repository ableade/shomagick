#include "shomatcher.hpp"
#include "kdtree.h"
#include "camera.h"
#include  "cudamatcher.hpp"
#include "RobustMatcher.h"
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "json.hpp"
#include <set>

using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::Mat;
using cv::ORB;
using cv::cuda::GpuMat;
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
using std::cerr;
using std::set;
using std::max;
using std::vector;
using std::string;
using json = nlohmann::json;

ShoMatcher::ShoMatcher(FlightSession flight, bool runCuda) : flight(flight), runCuda(runCuda)
, kd(nullptr)
, candidateImages()
, detector_(cv::ORB::create(4000))
, extractor_(cv::ORB::create(4000)) {
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        cerr << "CUDA device detected. Running CUDA \n";
        cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
        cudaEnabled = true;
    }
    if (cudaEnabled && runCuda) {
        //Set CUDA ORB detector
        detector_ = cv::cuda::ORB::create(4000);
        extractor_ = cv::cuda::ORB::create(4000);
    }
}
void ShoMatcher::getCandidateMatchesUsingSpatialSearch(double range)
{
    set<pair<string, string>> alreadyPaired;
    this->buildKdTree();
    auto imageSet = flight.getImageSet();
    for (const auto img: imageSet) {
        vector<string> matchSet;
        auto currentImageName = img.getFileName();
        void *result_set;

        double pt[] = {img.getMetadata().location.longitude, img.getMetadata().location.latitude };
        result_set = kd_nearest_range(static_cast<kdtree *>(kd), pt, range);
        vector<double> pos(this->dimensions);
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

int ShoMatcher::extractFeatures(bool resize) 
{
    //set feature process size to -1 to avoid resizing
    if (FEATURE_PROCESS_SIZE != -1 && resize) {
        auto maxSize = max(flight.getCamera().getHeight(), flight.getCamera().getWidth());
        int fx = flight.getCamera().getWidth() * FEATURE_PROCESS_SIZE / maxSize;
        int fy = flight.getCamera().getHeight() * FEATURE_PROCESS_SIZE / maxSize;
        flight.getCamera().setScaledHeight(fy);
        flight.getCamera().setScaledWidth(fx);
    }

    set<string> detected;
    if (!this->candidateImages.size())
        return 0;

    for (auto it = candidateImages.begin(); it != candidateImages.end(); it++)
    {
        if (detected.find(it->first) == detected.end() && _extractFeature(it->first, resize))
        {
            detected.insert(it->first);
        }

        for (auto _it = it->second.begin(); _it != it->second.end(); ++_it)
        {
            if (detected.find(*_it) == detected.end())
            {
                if (_extractFeature(*_it, resize))
                {
                    detected.insert(*_it);
                }
            }
        }
    }
    return detected.size();
}

bool ShoMatcher::_extractFeature(string fileName, bool resize)
{
    auto imageFeaturePath = flight.getImageFeaturesPath() / (fileName + ".yaml");
    auto modelimageNamePath = flight.getImageDirectoryPath() / (fileName);
    if (boost::filesystem::exists(imageFeaturePath)) {
        //Use existing file instead.
        cerr << "Using " << imageFeaturePath.string()<< " for features \n";
        return true;
    }
    
    Mat modelImg = imread(modelimageNamePath.string(), SHO_LOAD_COLOR_IMAGE_OPENCV_ENUM | SHO_LOAD_ANYDEPTH_IMAGE_OPENCV_ENUM);
    cout << "Model image name path is " << modelimageNamePath << "\n";
    cv::cvtColor(modelImg, modelImg, SHO_BGR2RGB);
    Mat featureImage = imread(modelimageNamePath.string(), SHO_GRAYSCALE);

    auto channels = modelImg.channels();

    if (modelImg.empty())
        return false;

    if (resize && featureImage.size().width > FEATURE_PROCESS_SIZE) {
        cv::resize(modelImg, modelImg, { flight.getCamera().getScaledWidth(), flight.getCamera().getScaledHeight()}, 0, 0, cv::INTER_AREA);
        cv::resize(featureImage, featureImage, { flight.getCamera().getScaledWidth(), flight.getCamera().getScaledHeight() }, 0, 0, cv::INTER_AREA);
    }

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Scalar> colors;
    cv::Mat descriptors;

    if (cudaEnabled && runCuda) {
        GpuMat cudaFeatureImg, cudaKeypoints,cudaDescriptors;
        cudaFeatureImg.upload(featureImage);
        auto cudaDetector =  detector_.dynamicCast<cv::cuda::ORB>();
        cudaDetector->detectAndCompute(cudaFeatureImg, cv::noArray(), keypoints, cudaDescriptors);
        cudaDescriptors.download(descriptors);
    }
    else {
        detector_->detectAndCompute(featureImage, cv::noArray(), keypoints, descriptors);
    }

    cout << "Extracted " << descriptors.rows << " points for  " << fileName << endl;

    for (auto &keypoint : keypoints) {
        if (channels == 1)
            colors.push_back(modelImg.at<uchar>(keypoint.pt));
        else if (channels == 3)
            colors.push_back(modelImg.at<Vec3b>(keypoint.pt));
    
        keypoint.pt = this->flight.getCamera().normalizeImageCoordinate(keypoint.pt);
    }
    return this->flight.saveImageFeaturesFile(fileName, keypoints, descriptors, colors);
}


void ShoMatcher::runRobustFeatureMatching()
{
    if (!this->candidateImages.size())
        return;

    RobustMatcher rmatcher;
    cv::Ptr<CUDARobustMatcher> cMatcher;
    if (cudaEnabled && runCuda) {
        cMatcher = cv::makePtr<CUDARobustMatcher>();
    }

    map<string, ImageFeatures> loadedFeatures;
    for (const auto&[queryImg, trainImages] : candidateImages) {
        vector<string> trainImageSet;
        auto queryImagePath = flight.getImageDirectoryPath() / queryImg;
        ImageFeatures queryFeaturesSet;
        try {
            queryFeaturesSet = loadedFeatures.at(queryImg);
        }
        catch (std::out_of_range e) {
            queryFeaturesSet = flight.loadFeatures(queryImg);
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
                trainFeaturesSet = flight.loadFeatures(trainImg);
                loadedFeatures[trainImg] = trainFeaturesSet;
            }
            vector<DMatch> matches;
            
            if (cudaEnabled && runCuda) {
                cMatcher->robustMatch(queryFeaturesSet.descriptors, trainFeaturesSet.descriptors, matches);
            }
            else {
                rmatcher.robustMatch(queryFeaturesSet.descriptors, trainFeaturesSet.descriptors, matches);
            }

            int trainIndex = this->flight.getImageIndex(trainImg);
            for (size_t i = 0; i < matches.size(); i++)
            {
                //Update train index so we know what image we matched against when we are running the tracking pipeline
                matches[i].imgIdx = trainIndex;
            }
            matchSet[trainImg] = matches;
            cout << queryImg << " - " << trainImg << " has " << matches.size() << "candidate matches" << endl;
        }
        this->flight.saveMatches(queryImg, matchSet);
    }
}

void ShoMatcher::buildKdTree()
{
    kd = kd_create(this->dimensions);
    for (auto &img : this->flight.getImageSet())
    {
        auto pos = vector<double>{ img.getMetadata().location.longitude, img.getMetadata().location.latitude };
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree *>(kd), pos.data(), dt) == 0);
    }

}

map<string, std::vector<string>> ShoMatcher::getCandidateImages() const
{
    return this->candidateImages;
}

void ShoMatcher::plotMatches(string img1, string img2) const {
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

    const auto frameName = img1 + " - " + img2 + " matches";
    cv::namedWindow(frameName, cv::WINDOW_NORMAL);
    imshow(frameName, imageMatches);
}

