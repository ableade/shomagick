#include "image.hpp"
#include <string>
#include <fstream>
#include "utilities.h"
#include "bootstrap.h"
#include <boost/algorithm/string.hpp>
#include "sensordata.h"
using std::string;

double toRadian(double deg)
{
    return deg * M_PI / DEG;
}

string parseFileNameFromPath(std::string path)
{
    return boost::filesystem::path{ path }.filename().string();
}


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
    ImageMetadata imageExif{};
    auto image = Exiv2::ImageFactory::open(imagePath);
    assert(image.get() != 0);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty())
    {
        std::cerr << "Warning! Missing exif data \n";
    }
    imageExif.height = image->pixelHeight();
    imageExif.width = image->pixelWidth();
    const auto loc = _extractCoordinatesFromExif(exifData);
    const auto[make, model] = _extractMakeAndModelFromExif(exifData);
    const auto orientation = _extractOrientationFromExif(exifData);
    _extractLensModel(imageExif, exifData);
    _extractExifWidthAndHeight(imagePath, imageExif, exifData);
    imageExif.location = loc;
    imageExif.cameraMake = make;
    imageExif.cameraModel = model;
    imageExif.orientation = orientation;
    _extractFocalMetadata(imageExif, exifData);
    _extractFocalLengthFromExifData(imageExif, exifData);
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
    return { longitudeRef * longitude, latitudeRef * latitude, altitude, dop, false };
}

void Img::_extractFocalLengthFromExifData(ImageMetadata& metadata, Exiv2::ExifData exifData)
{
    if (metadata.focal35 > 0) {
        metadata.focalRatio = metadata.focal35 / 36.0;
        return;
    }

    if (metadata.sensorWidth != 0 && metadata.focal != 0) {
        metadata.focalRatio = metadata.focal / metadata.sensorWidth;
        metadata.focal35 = 36.0 * metadata.focalRatio;
        return;
    }

    //Default focal 35 and focal ratio to zero.
    metadata.focal35 = 0;
    metadata.focalRatio = 0;
}

void Img::_extractProjectionTypeFromExif(ImageMetadata & metadata, Exiv2::ExifData exifData)
{
    metadata.projectionType = "perspective";
}

Img::CameraMakeAndModel Img::_extractMakeAndModelFromExif(Exiv2::ExifData exifData)
{
    string cameraModel = "";
    string cameraMake = "";
    auto makeKey = Exiv2::ExifKey("Exif.Image.Make");
    auto modelKey = Exiv2::ExifKey("Exif.Image.Model");


    if ((exifData.findKey(makeKey)) != exifData.end()) {
        cameraMake = exifData.findKey(makeKey)->toString();
    }
    if (exifData.findKey(modelKey) != exifData.end()) {
        cameraModel = exifData.findKey(modelKey)->toString();
    }
    boost::to_lower(cameraMake);
    boost::to_lower(cameraModel);
    return make_tuple(cameraMake, cameraModel);
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

void Img::_extractLensModel(ImageMetadata & metaData, Exiv2::ExifData imageExifData)
{
    const auto lensKey = Exiv2::ExifKey("Exif.Photo.LensModel");
    const auto lensIt = imageExifData.findKey(lensKey);
    if (lensIt != imageExifData.end()) {
        metaData.lensModel = lensIt->getValue()->toString();
    }
}

void Img::_extractExifWidthAndHeight(string imagePath, ImageMetadata & metadata, Exiv2::ExifData imageExifData)
{
    auto heightKey = Exiv2::ExifKey("Exif.Image.ImageLength");
    auto widthKey = Exiv2::ExifKey("Exif.Image.ImageWidth");

    if (imageExifData.findKey(heightKey) == imageExifData.end())
        heightKey = Exiv2::ExifKey("Exif.Photo.PixelYDimension");
    if(imageExifData.findKey(widthKey) == imageExifData.end())
        widthKey = Exiv2::ExifKey("Exif.Photo.PixelXDimension");

    auto hIt = imageExifData.findKey(heightKey);
    auto wIt = imageExifData.findKey(widthKey);

    if (hIt != imageExifData.end() && wIt != imageExifData.end()) {
        metadata.height = hIt->getValue()->toLong();
        metadata.width = wIt->getValue()->toLong();
    } 
}

void Img::_extractFocalMetadata(ImageMetadata & metadata, Exiv2::ExifData imageExifData)
{
    //Extract the sensor width if available
    auto fPXKey = Exiv2::ExifKey("Exif.Image.FocalPlaneXResolution");
    auto fPrKey = Exiv2::ExifKey("Exif.Image.FocalPlaneResolutionUnit");
    auto focal35key = Exiv2::ExifKey("Exif.Photo.FocalLengthIn35mmFilm");
    auto focalKey = Exiv2::ExifKey("Exif.Image.FocalLength");

    if (imageExifData.findKey(fPXKey) != imageExifData.end() &&
        imageExifData.findKey(fPrKey) != imageExifData.end()) {
        auto resolutionUnit = imageExifData.findKey(fPrKey)->getValue()->toLong();
        float mmUnit = 0;

        if (resolutionUnit == 2)
            mmUnit = 25.4; //1 inch is 25.4 millimeters

        else if (resolutionUnit == 3)
            mmUnit = 10; // I centimeter is 10 millimeters

        if (mmUnit != 0) {
            //Calculate sensor width
            auto pixelsPerUnit = imageExifData.findKey(fPXKey)->getValue()->toFloat();
            auto unitsPerPixel = 1 / pixelsPerUnit;
            metadata.sensorWidth = metadata.width * unitsPerPixel * mmUnit;
        }
    }
    else {
        _extractSensorWidthFromDB(metadata, imageExifData);
    }

    if (imageExifData.findKey(focal35key) != imageExifData.end())
        metadata.focal35 = imageExifData.findKey(focal35key)->getValue()->toFloat();
    if (imageExifData.findKey(focalKey) != imageExifData.end())
        metadata.focal = imageExifData.findKey(focalKey)->getValue()->toFloat();
    else {
        //Focal key has multiple exif tag definitions
        focalKey = Exiv2::ExifKey("Exif.Photo.FocalLength");
        if (imageExifData.findKey(focalKey) != imageExifData.end()) {
            metadata.focal = imageExifData.findKey(focalKey)->getValue()->toFloat();
        }
    }
}

void Img::_extractSensorWidthFromDB(ImageMetadata & metadata, Exiv2::ExifData imageExifData)
{
    auto make = metadata.cameraMake;
    auto model = metadata.cameraModel;
    boost::erase_all(model, make);
    boost::trim(model);
    boost::trim(make);
    auto sensorId = make + ' ' + model;

    if (sensorSizes.find(sensorId) != sensorSizes.end()) {
        metadata.sensorWidth = sensorSizes.at(sensorId);
    }
}

Img::Img(string imageFileName) : imageFileName(imageFileName) {

}