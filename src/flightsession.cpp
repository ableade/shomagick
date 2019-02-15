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

FlightSession::FlightSession() : imageData(), imageDirectory(), imageDirectoryPath(), imageFeaturesPath(),
imageTracksPath(), camera()
{

}

FlightSession::FlightSession(string imageDirectory, string calibrationFile) : imageData(), imageDirectory(imageDirectory), imageDirectoryPath(), imageFeaturesPath(),
imageTracksPath(), camera()
{
    cerr << "Image directory is " << imageDirectory << endl;
    vector<directory_entry> v;
    assert(is_directory(imageDirectory));
    this->imageDirectoryPath = path(imageDirectory);

    this->imageFeaturesPath = this->imageDirectoryPath / "features";
    this->imageMatchesPath = this->imageDirectoryPath / "matches";
    this->imageTracksPath = this->imageDirectoryPath / "tracks";
    cout << "Creating directory " << this->imageFeaturesPath.string() << endl;
    boost::filesystem::create_directory(this->imageFeaturesPath);
    cout << "Creating directory " << this->imageMatchesPath.string() << endl;
    boost::filesystem::create_directory(this->imageMatchesPath);
    cout << "Creating directory " << this->imageTracksPath.string() << endl;
    boost::filesystem::create_directory(this->imageTracksPath);
    copy_if(
        directory_iterator(imageDirectory),
        directory_iterator(),
        back_inserter(v),
        [](const directory_entry &e) {
        return is_regular_file(e);
    });
    for (auto entry : v)
    {
        Img img;
        img.fileName = parseFileNameFromPath(entry.path().string());
       
        auto metadata = _extractExifFromImage(entry.path().string());
        img.metadata = metadata;
        this->imageData.push_back(img);
    }
    if (!calibrationFile.empty()) {
        this->camera = Camera::getCameraFromCalibrationFile(calibrationFile);
    }
    cout << "Found " << this->imageData.size() << " usable images" << endl;
}

ImageMetadata FlightSession::_extractExifFromImage(std::string imagePath) const
{
    ImageMetadata imageExif;
    auto image = Exiv2::ImageFactory::open(imagePath);
    assert(image.get() != 0);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty())
    {
        std::string error(imagePath);
        error += ": No Exif data found in the file";
        throw Exiv2::Error(Exiv2::ErrorCode::kerGeneralError, error);
    }
    const auto loc = _extractCoordinates(exifData);
    const auto[make, model] = _extractMakeAndModel(exifData);

    return imageExif;
}

Location FlightSession::_extractCoordinates(const Exiv2::ExifData exifData) const
{
    Exiv2::ExifData::const_iterator end = exifData.end();
    Exiv2::Value::UniquePtr latV = Exiv2::Value::create(Exiv2::signedRational);
    Exiv2::Value::UniquePtr longV = Exiv2::Value::create(Exiv2::signedRational);
    auto longitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude");
    auto latitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude");
    auto latPos = exifData.findKey(latitudeKey);
    auto longPos = exifData.findKey(longitudeKey);
    if (latPos == exifData.end() || longPos == exifData.end())
        throw Exiv2::Error(Exiv2::ErrorCode::kerInvalidKey, "Key not found");
    // Get a pointer to a copy of the value
    latV = latPos->getValue();
    longV = longPos->getValue();
    auto latitude = latV->toFloat() + (latV->toFloat(1) / 60.0) + (latV->toFloat(2) / 3600.0);
    auto longitude = longV->toFloat() + (longV->toFloat(1) / 60.0) + (longV->toFloat(2) / 3600.0);

    return {longitude, latitude, 0};
}

FlightSession::CameraMakeAndModel FlightSession::_extractMakeAndModel(const Exiv2::ExifData exifData) const
{
    auto makeKey = Exiv2::ExifKey("Exif.Image.Make");
    auto modelKey = Exiv2::ExifKey("Exif.Image.Model");

    Exiv2::Value::UniquePtr makeV = Exiv2::Value::create(Exiv2::string);
    Exiv2::Value::UniquePtr modelV = Exiv2::Value::create(Exiv2::string);

    return make_tuple(makeV->toString(), modelV->toString());

}

vector<Img> FlightSession::getImageSet() const
{
    return this->imageData;
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
    for (size_t i = 0; i < this->imageData.size(); ++i)
    {
        if (this->imageData[i].fileName == imageName)
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

void FlightSession::setCamera(Camera camera)
{
    this->camera = camera;
}
