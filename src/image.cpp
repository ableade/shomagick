#include "image.hpp"
#include <string>
#include <fstream>
#include "utilities.h"
using std::string;

const ImageMetadata & Img::getMetadata() const
{
    return metadata;
}

const std::string & Img::getFileName() const
{
    return imageFileName;
}

ImageMetadata& Img::getMetadata()
{
    return metadata;
}

ImageMetadata Img::extractExifFromImage(std::string imagePath)
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
    const auto loc = _extractCoordinatesFromExif(exifData);
    const auto[make, model] = _extractMakeAndModelFromExif(exifData);
    const auto orientation = _extractOrientationFromExif(exifData);

    imageExif.location = loc;
    imageExif.cameraMake = make;
    imageExif.cameraModel = model;
    imageExif.orientation = orientation;
    return imageExif;
}

void Img::extractExifFromFile(std::string imageExifFile, ImageMetadata& imgMetadata)
{
    assert(boost::filesystem::exists(imageExifFile));
    std::ifstream exifFile(imageExifFile, std::ios::in);
    boost::archive::text_iarchive ar(exifFile);
    ar & imgMetadata;
}

Location Img::_extractCoordinatesFromExif(Exiv2::ExifData exifData)
{
    auto latitudeRef = 1;
    auto longitudeRef = 1;
    Exiv2::ExifData::const_iterator end = exifData.end();
    Exiv2::Value::UniquePtr latV = Exiv2::Value::create(Exiv2::signedRational);
    Exiv2::Value::UniquePtr longV = Exiv2::Value::create(Exiv2::signedRational);
    Exiv2::Value::UniquePtr altV = Exiv2::Value::create(Exiv2::signedRational);


    auto longitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude");
    auto latitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude");
    auto altitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSAltitude");
    auto latitudeRefKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLatitudeRef");
    auto longitudeRefKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLongitudeRef");

    const auto latPos = exifData.findKey(latitudeKey);
    const auto longPos = exifData.findKey(longitudeKey);
    const auto altPos = exifData.findKey(altitudeKey);
    const auto latRefPos = exifData.findKey(latitudeRefKey);
    const auto lonRefPos = exifData.findKey(longitudeRefKey);

    if (latRefPos != exifData.end()) {
        const auto latRef = latRefPos->getValue()->toString();
        if (latRef == "S")
            latitudeRef = -1;
    }

    if (lonRefPos != exifData.end()) {
        const auto lonRef = lonRefPos->getValue()->toString();
        if (lonRef == "W")
            longitudeRef = -1;
    }

    if (latPos == exifData.end() || longPos == exifData.end() || altPos == exifData.end())
        return {};

    // Get a pointer to a copy of the value
    latV = latPos->getValue();
    longV = longPos->getValue();
    altV = altPos->getValue();
    altV = altPos->getValue();
    auto latitude = latV->toFloat() + (latV->toFloat(1) / 60.0) + (latV->toFloat(2) / 3600.0);
    auto longitude = longV->toFloat() + (longV->toFloat(1) / 60.0) + (longV->toFloat(2) / 3600.0);
    auto altitude = altV->toFloat();
    auto dop = _extractDopFromExif(exifData);

    // TODO  check the alttude value that is being parsed.
    return {longitudeRef * longitude, latitudeRef * latitude, altitude, dop, false };
}

Img::CameraMakeAndModel Img::_extractMakeAndModelFromExif(Exiv2::ExifData exifData)
{
    auto makeKey = Exiv2::ExifKey("Exif.Image.Make");
    auto modelKey = Exiv2::ExifKey("Exif.Image.Model");

    Exiv2::Value::UniquePtr makeV = Exiv2::Value::create(Exiv2::string);
    Exiv2::Value::UniquePtr modelV = Exiv2::Value::create(Exiv2::string);

    auto cameraMake = exifData.findKey(makeKey);
    auto cameraModel = exifData.findKey(modelKey);
    makeV = cameraMake->getValue();
    modelV = cameraModel->getValue();

    return make_tuple(makeV->toString(), modelV->toString());
}

double Img::_extractDopFromExif(Exiv2::ExifData imageExifData)
{
    const auto dop = 0.0;
    const auto dopKey = Exiv2::ExifKey("Exif.GPSInfo.GPSDOP");
    const auto dopIt = imageExifData.findKey(dopKey);
    if (dopIt != imageExifData.end()) {
        const auto dopVal = dopIt->getValue();
        return dopVal->toFloat();
    }
    return dop;
}

int Img::_extractOrientationFromExif(Exiv2::ExifData imageExifData)
{
    const auto orientation = 1;
    const auto orientationKey = Exiv2::ExifKey("Exif.Thumbnail.Orientation");
    const auto orientationIt = imageExifData.findKey(orientationKey);
    if (orientationIt != imageExifData.end()) {
        const auto orientationVal = orientationIt->getValue();
        return orientationVal->toLong();
    }
    return orientation;
}

Img::Img(string imageFileName) : imageFileName(imageFileName) {

}
