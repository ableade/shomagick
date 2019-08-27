#include "flightsession.h"
#include "exiv2/exiv2.hpp"
#include <iostream>
#include <fstream>
#include "utilities.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem.hpp>
#include "bootstrap.h"

using namespace boost::filesystem;
using cv::DMatch;
using cv::FileNode;
using cv::FileNodeIterator;
using cv::KeyPoint;
using cv::Mat;
using cv::Scalar;
using std::cout;
using std::map;
using std::vector;
using std::cerr;
using std::string;
using std::ofstream;
using std::endl;
using std::ios;

FlightSession::FlightSession() : imageSet(), flightSessionDirectory_(), imageDirectoryPath_(), imageFeaturesPath_(),
imageTracksPath_(), camera_(), referenceLLA_()
{

}

FlightSession::FlightSession(string flightSessionDirectory, string calibrationFile) : imageSet(), flightSessionDirectory_(flightSessionDirectory), imageDirectoryPath_(), imageFeaturesPath_(),
imageTracksPath_(), camera_(), referenceLLA_()
{
    //Remove trailing slash if present at the end to avoid unexpected bugs with boost file system paths
    auto lastChar = flightSessionDirectory.at(flightSessionDirectory.size() - 1);
    if (lastChar == '/' || lastChar == '\\') {
        flightSessionDirectory.erase(flightSessionDirectory.size() - 1);
    }

    cerr << "Flight session directory is " << flightSessionDirectory << endl;
    vector<directory_entry> v;
    assert(is_directory(flightSessionDirectory));
    imageDirectoryPath_ = path(flightSessionDirectory) / "images";
    assert(is_directory(imageDirectoryPath_));
    imageFeaturesPath_ = flightSessionDirectory / "sho-features";
    imageMatchesPath_ = flightSessionDirectory_ / "sho-matches";
    imageTracksPath_ = flightSessionDirectory_ / "sho-tracks";
    exifPath_ = flightSessionDirectory_ / "sho-exif";
    undistortedImagesPath_ = flightSessionDirectory_ / "sho-undisorted";
    reconstructionPaths_ = flightSessionDirectory_ / "sho-reconstructions";

    const vector <boost::filesystem::path> allPaths{ imageFeaturesPath_, imageMatchesPath_,
    imageTracksPath_, exifPath_, undistortedImagesPath_ , reconstructionPaths_};

    
    for (const auto path : allPaths) {
        cout << "Creating directory " << path.string() << endl;
        boost::filesystem::create_directory(path);
    }
    
    copy_if(
        directory_iterator(imageDirectoryPath_),
        directory_iterator(),
        back_inserter(v),
        [](const directory_entry &e) {
        return is_regular_file(e);
    });
    for (auto entry : v)
    {
        if (entry.path().extension().string() == ".jpg" || entry.path().extension().string() == ".png") {
            const auto imageFileName = parseFileNameFromPath(entry.path().string());
            const auto imageExifPath = getImageExifPath() / (imageFileName + ".dat");
            ImageMetadata metadata;
            if (exists(imageExifPath)) {
                Img::extractExifFromFile(imageExifPath.string(), metadata);
            } else{
                metadata = Img::extractExifFromImage(entry.path().string());
                saveImageExifFile(imageFileName, metadata);
            }
            Img img(imageFileName, metadata);
            if (metadata.location.isEmpty) {
                gpsDataPresent_ = false;
            }
            imageSet.push_back(img);
        }
    }
    if (!calibrationFile.empty()) {
        assert(exists(calibrationFile));
        camera_ = Camera::getCameraFromCalibrationFile(calibrationFile);
    }
    inventReferenceLLA();
    cout << "Found " << this->imageSet.size() << " usable images" << endl;
}

std::string FlightSession::_extractProjectionTypeFromExif(Exiv2::ExifData exifData) const
{
    return std::string("perspective");
}

vector<Img> FlightSession::getImageSet() const
{
    return this->imageSet;
}

const path FlightSession::getImageDirectoryPath() const
{
    return this->imageDirectoryPath_;
}

const path FlightSession::getImageFeaturesPath() const
{
    return this->imageFeaturesPath_;
}

const path FlightSession::getImageMatchesPath() const
{
    return imageMatchesPath_;
}

const boost::filesystem::path FlightSession::getImageExifPath() const
{
    return exifPath_;
}

/*
bool FlightSession::saveTracksFile(std::map <int, std::vector <int>> tracks) {
    if (! tracks.size())
        return false;

    auto imageTracksFile = this->getImageTracksPath() / "tracks.yaml";
    cv::FileStorage fs(imageTracksFile.string(), cv::FileStorage::WRITE);
    fs << "Tracks" << tracks;
    return boost::filesystem::exists(imageTracksFile);
}
*/

const path FlightSession::getImageTracksPath() const
{
    return imageTracksPath_;
}

const boost::filesystem::path FlightSession::getUndistortedImagesDirectoryPath() const
{
    return undistortedImagesPath_;
}

const boost::filesystem::path FlightSession::getReconstructionsPath() const
{
    return reconstructionPaths_;
}

int FlightSession::getImageIndex(string imageName) const
{
    for (size_t i = 0; i < imageSet.size(); ++i)
    {
        if (imageSet[i].getFileName() == imageName)
        {
            return i;
        }
    }
    return -1;
}
bool FlightSession::saveImageFeaturesFile(string imageName, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors,
    const std::vector<cv::Scalar> &colors)
{
    auto imageFeaturePath = getImageFeaturesPath() / (imageName + ".yaml");
    if (!boost::filesystem::exists(imageFeaturePath))
    {
        cv::FileStorage file(imageFeaturePath.string(), cv::FileStorage::WRITE);
        file << "Keypoints" << keypoints;
        file << "Descriptors" << descriptors;
        file << "Colors" << colors;
        file.release();
    }
    return boost::filesystem::exists(imageFeaturePath);
}

bool FlightSession::saveImageExifFile(std::string imageName, ImageMetadata imageExif)
{
    auto imageExifPath = getImageExifPath() / (imageName + ".dat");
    std::ofstream exifFile(imageExifPath.string(), ios::binary);
    boost::archive::text_oarchive ar(exifFile);
    ar & imageExif;
    return exists(imageExifPath);
}

bool FlightSession::saveMatches(string fileName, const std::map<string, vector<cv::DMatch>>& matches)
{
    auto imageMatchesPath = getImageMatchesPath() / (fileName + ".yaml");
    cout << "Writing file " << imageMatchesPath.string() << endl;
    cv::FileStorage fs(imageMatchesPath.string(), cv::FileStorage::WRITE);
    fs << "MatchCount" << (int)matches.size();
    fs << "candidateImageMatches" << "[";
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        fs << "{" << "imageName" << it->first;
        fs << "matches" << it->second << "}";
    }
    fs << "]";
    fs.release();
    return boost::filesystem::exists(imageMatchesPath);
}

map<string, vector<DMatch>> FlightSession::loadMatches(string fileName) const
{
    map<string, vector<DMatch>> allPairMatches;

    auto imageMatchesPath = getImageMatchesPath() / (fileName + ".yaml");
    cv::FileStorage fs(imageMatchesPath.string(), cv::FileStorage::READ);
    FileNode cMatches = fs["candidateImageMatches"];
    FileNodeIterator it = cMatches.begin(), it_end = cMatches.end();
    for (; it != it_end; ++it)
    {
        vector<DMatch> pairwiseMatches;
        (*it)["matches"] >> pairwiseMatches;
        string matchImage = (string)(*it)["imageName"];
        allPairMatches.insert(make_pair(matchImage, pairwiseMatches));
    }
    return allPairMatches;
}

ImageFeatures FlightSession::loadFeatures(string imageName) const
{
    auto imageFeaturePath = getImageFeaturesPath() / (imageName + ".yaml");
    cv::FileStorage fs(imageFeaturePath.string(), cv::FileStorage::READ);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    vector<Scalar> colors;
    fs["Keypoints"] >> keypoints;
    fs["Descriptors"] >> descriptors;
    fs["Colors"] >> colors;

    return { keypoints, descriptors, colors };
}

const Camera& FlightSession::getCamera() const {
    return camera_;
}

Camera & FlightSession::getCamera()
{
    return camera_;
}

void FlightSession::setCamera(Camera camera)
{
    this->camera_ = camera;
}

namespace
{

} //namespace

void FlightSession::inventReferenceLLA()
{
    auto lat = 0.0;
    auto lon = 0.0;
    auto alt = 0.0;
    auto wAlt = 0.0;
    auto wLat = 0.0;
    auto wLon = 0.0;
    const auto defaultDop = 15;
    for (const auto img : imageSet) {
        const auto dop = (img.getMetadata().location.dop != 0.0) ? img.getMetadata().location.dop : defaultDop;
        const auto w = 1.0 / std:: max(0.01, dop);
        lat += w * img.getMetadata().location.latitude;
        lon += w * img.getMetadata().location.longitude;
        wLat += w;
        wLon += w;
        alt += w * img.getMetadata().location.altitude;
        wAlt += w;
    }
    lat /= wLat;
    lon /= wLon;
    alt /= wAlt;

    referenceLLA_ = { {"alt", 0}, {"lat",  lat}, {"lon" , lon} };
}

const std::map<std::string, double>& FlightSession::getReferenceLLA() const
{
    return referenceLLA_;
}

bool FlightSession::hasGps()
{
    return gpsDataPresent_;
}

void FlightSession::undistort()
{
    for (auto img : imageSet) {
        auto imagePath = imageDirectoryPath_ / img.getFileName();
        Mat distortedImage = cv::imread(imagePath.string(),
            SHO_LOAD_COLOR_IMAGE_OPENCV_ENUM |
            SHO_LOAD_ANYDEPTH_IMAGE_OPENCV_ENUM);
        if (!distortedImage.data)
            continue;
        Mat undistortedImage;
        cv::undistort(distortedImage, 
            undistortedImage, 
            camera_.getKMatrix(), 
            camera_.getDistortionMatrix());
        auto undistortedImagePath = undistortedImagesPath_ / img.getFileName();
        undistortedImagePath.replace_extension("png");
        cv::imwrite(undistortedImagePath.string(), undistortedImage);
    }
   
}
