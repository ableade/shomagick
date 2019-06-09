#include "flightsession.h"
#include "exiv2/exiv2.hpp"
#include <iostream>
#include <fstream>
#include "utilities.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

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
using std::ios;

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
    imageDirectoryPath = path(imageDirectory);

    imageFeaturesPath = imageDirectoryPath / "features";
    imageMatchesPath = imageDirectoryPath / "matches";
    imageTracksPath = imageDirectoryPath / "tracks";
    exifPath = imageDirectoryPath / "exif";
    undistortedImagesPath = imageDirectoryPath / "undisorted";

    const vector <boost::filesystem::path> allPaths{ imageFeaturesPath, imageMatchesPath,
    imageTracksPath, exifPath, undistortedImagesPath };

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
                gpsDataPresent = false;
            }
            imageSet.push_back(img);
        }
    }
    if (!calibrationFile.empty()) {
        assert(exists(calibrationFile));
        camera = Camera::getCameraFromCalibrationFile(calibrationFile);
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
    return imageMatchesPath;
}

const boost::filesystem::path FlightSession::getImageExifPath() const
{
    return exifPath;
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
    return imageTracksPath;
}

const boost::filesystem::path FlightSession::getUndistortedImagesDirectoryPath() const
{
    return undistortedImagesPath;
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
    boost::filesystem::ofstream exifFile(imageExifPath, ios::binary);
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
    return camera;
}

Camera & FlightSession::getCamera()
{
    return camera;
}

void FlightSession::setCamera(Camera camera)
{
    this->camera = camera;
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

    referenceLLA = { {"alt", 0}, {"lat",  lat}, {"lon" , lon} };
}

const std::map<std::string, double>& FlightSession::getReferenceLLA() const
{
    return referenceLLA;
}

bool FlightSession::hasGps()
{
    return gpsDataPresent;
}

void FlightSession::undistort()
{
    for (auto img : imageSet) {
        auto imagePath = imageDirectoryPath / img.getFileName();
        Mat distortedImage = cv::imread(imagePath.string(),
            SHO_LOAD_COLOR_IMAGE_OPENCV_ENUM |
            SHO_LOAD_ANYDEPTH_IMAGE_OPENCV_ENUM);
        if (!distortedImage.data)
            continue;
        Mat undistortedImage;
        cv::undistort(distortedImage, 
            undistortedImage, 
            camera.getKMatrix(), 
            camera.getDistortionMatrix());
        auto undistortedImagePath = undistortedImagesPath / img.getFileName();
        undistortedImagePath.replace_extension("png");
        cv::imwrite(undistortedImagePath.string(), undistortedImage);
    }
   
}
