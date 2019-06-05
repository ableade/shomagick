#pragma once

#include "image.hpp"
#include "camera.h"
#include <boost/filesystem.hpp>
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

private:
    std::vector<Img> imageSet;
    std::string imageDirectory;
    boost::filesystem::path imageDirectoryPath;
    boost::filesystem::path imageFeaturesPath;
    boost::filesystem::path imageMatchesPath;
    boost::filesystem::path imageTracksPath;
    boost::filesystem::path exifPath;
    boost::filesystem::path undistortedImagesPath;
    Camera camera;
    std::map<std::string, double> referenceLLA;
    // TODO Implement unimplemented functions in flightsession class
    std::string _extractProjectionTypeFromExif(Exiv2::ExifData exifData) const;
    bool gpsDataPresent = true;

public:
    FlightSession();
    FlightSession(std::string imageDirectory, std::string calibFile = std::string());
    std::vector<Img> getImageSet() const;
    const boost::filesystem::path getImageDirectoryPath() const;
    const boost::filesystem::path getImageFeaturesPath() const;
    const boost::filesystem::path getImageMatchesPath() const;
    const boost::filesystem::path getImageExifPath() const;
    const boost::filesystem::path getImageTracksPath() const;
    const boost::filesystem::path getUndistortedImagesDirectoryPath()const;
    bool saveTracksFile(std::map<int, std::vector<int>> tracks);
    int getImageIndex(std::string imageName) const;
    std::map<std::string, std::vector<cv::DMatch>> loadMatches(std::string fileName) const;
    bool saveImageFeaturesFile(
        std::string imageName, 
        const std::vector<cv::KeyPoint> &keypoints, 
        const cv::Mat& descriptors,
        const std::vector<cv::Scalar>& colors
    );
    bool saveImageExifFile(std::string imageName, ImageMetadata imageExif);
    bool saveMatches(std::string fileName, const std::map<std::string, std::vector<cv::DMatch>>& matches);
    ImageFeatures loadFeatures(std::string imageName) const;
    const Camera& getCamera() const;
    Camera& getCamera();
    void setCamera(Camera camera);
    void inventReferenceLLA();
    const std::map<std::string, double>& getReferenceLLA() const;
    bool hasGps();
    void undistort();
};
