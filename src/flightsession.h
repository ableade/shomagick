#pragma once

#include "image.hpp"
#include "camera.h"
#include <boost/filesystem.hpp>
#include <exiv2/exiv2.hpp>
#include <map>
#include <vector>
#include <string>

class ImageFeatures {
public:
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Scalar> colors;
    std::vector<cv::KeyPoint> getKeypoints() const { return keypoints; }
    cv::Mat getDescriptors() const { return descriptors; }
};

class FlightSession
{
public:
    using CameraMake = std::string;
    using CameraModel = std::string;
    using CameraMakeAndModel = std::tuple<CameraMake, CameraModel>;

private:
    std::vector<Img> imageData;
    std::string imageDirectory;
    boost::filesystem::path imageDirectoryPath;
    boost::filesystem::path imageFeaturesPath;
    boost::filesystem::path imageMatchesPath;
    boost::filesystem::path imageTracksPath;
    Camera camera;
    ImageMetadata _extractExifFromImage(std::string imageName) const;
    Location _extractCoordinates(Exiv2::ExifData exifData) const;
    CameraMakeAndModel _extractMakeAndModel(Exiv2::ExifData exifData) const;

public:
    FlightSession();
    FlightSession(std::string imageDirectory, std::string calibFile = std::string());
    std::vector<Img> getImageSet() const;
    const boost::filesystem::path getImageDirectoryPath() const;
    const boost::filesystem::path getImageFeaturesPath() const;
    const boost::filesystem::path getImageMatchesPath() const;
    const boost::filesystem::path getImageTracksPath() const;
    bool saveTracksFile(std::map<int, std::vector<int>> tracks);
    int getImageIndex(std::string imageName) const;
    std::map<std::string, std::vector<cv::DMatch>> loadMatches(std::string fileName) const;
    bool saveImageFeaturesFile(
        std::string imageName, 
        const std::vector<cv::KeyPoint> &keypoints, 
        const cv::Mat& descriptors,
        const std::vector<cv::Scalar>& colors
    );
    bool saveMatches(std::string fileName, const std::map<std::string, std::vector<cv::DMatch>>& matches);
    ImageFeatures loadFeatures(std::string imageName) const;
    const Camera& getCamera() const;
    void setCamera(Camera camera);
};
