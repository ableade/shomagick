#include "flightsession.h"
#include "exiv2/exiv2.hpp"
#include <iostream>

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
using std::endl;

FlightSession::FlightSession() : imageSet(), imageDirectory(), imageDirectoryPath(), imageFeaturesPath(),
imageTracksPath(), camera(), referenceLLA()
{

}

FlightSession::FlightSession(string imageDirectory, string calibrationFile) : imageSet(), imageDirectory(imageDirectory), imageDirectoryPath(), imageFeaturesPath(),
imageTracksPath(), camera(), referenceLLA()
{
    cerr << "Image directory is " << imageDirectory << endl;
    vector<directory_entry> v;
    assert(is_directory(imageDirectory));
    this->imageDirectoryPath = path(imageDirectory);

    this->imageFeaturesPath = this->imageDirectoryPath / "features";
    this->imageMatchesPath = this->imageDirectoryPath / "matches";
    this->imageTracksPath = this->imageDirectoryPath / "tracks";
    this->exifPath = this->imageDirectoryPath / "exif";

    const vector <boost::filesystem::path> allPaths{ imageFeaturesPath, imageMatchesPath,
    imageTracksPath, exifPath };

    for (const auto path : allPaths) {
        cout << "Creating directory " << path.string() << endl;
        boost::filesystem::create_directory(path);
    }
    
    copy_if(
        directory_iterator(imageDirectory),
        directory_iterator(),
        back_inserter(v),
        [](const directory_entry &e) {
        return is_regular_file(e);
    });
    for (auto entry : v)
    {
        if (entry.path().extension().string() == ".jpg" || entry.path().extension().string() == ".png") {
            Img img(entry.path().string());
            this->imageSet.push_back(img);
        }
    }
    if (!calibrationFile.empty()) {
        this->camera = Camera::getCameraFromCalibrationFile(calibrationFile);
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
    return this->imageDirectoryPath;
}

const path FlightSession::getImageFeaturesPath() const
{
    return this->imageFeaturesPath;
}

const path FlightSession::getImageMatchesPath() const
{
    return this->imageMatchesPath;
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
    return this->imageTracksPath;
}

int FlightSession::getImageIndex(string imageName) const
{
    for (size_t i = 0; i < this->imageSet.size(); ++i)
    {
        if (this->imageSet[i].getFileName() == imageName)
        {
            return i;
        }
    }
    return -1;
}
bool FlightSession::saveImageFeaturesFile(string imageName, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors,
    const std::vector<cv::Scalar> &colors)
{
    auto imageFeaturePath = this->getImageFeaturesPath() / (imageName + ".yaml");
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

bool FlightSession::saveMatches(string fileName, const std::map<string, vector<cv::DMatch>>& matches)
{
    auto imageMatchesPath = this->getImageMatchesPath() / (fileName + ".yaml");
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
    cout << "Loading matches for " << fileName << endl;
    map<string, vector<DMatch>> allPairMatches;

    auto imageMatchesPath = this->getImageMatchesPath() / (fileName + ".yaml");
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
    cout << "Loading features for " << imageName << endl;
    auto imageFeaturePath = this->getImageFeaturesPath() / (imageName + ".yaml");
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
    return this->camera;
}

Camera & FlightSession::getCamera()
{
    return camera;
}

void FlightSession::setCamera(Camera camera)
{
    this->camera = camera;
}

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
        auto dop = (img.getMetadata().location.dop != 0.0) ? img.getMetadata().location.dop : defaultDop;
        auto w = 1.0 / std:: max(0.01, dop);
        lat += img.getMetadata().location.latitude;
        lon += img.getMetadata().location.longitude;
        wLat += w;
        wLon += w;
        alt += img.getMetadata().location.altitude;
        wAlt += w;
        lat /= wLat;
        lon /= wLon;
        alt /= wAlt;
    }
    referenceLLA = { {"alt", alt}, {"lat",  lat}, {"lon" , lon} };
}

const std::map<std::string, double>& FlightSession::getReferenceLLA() const
{
    return referenceLLA;
}
