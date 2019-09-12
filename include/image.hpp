#pragma once

#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <exiv2/exiv2.hpp>
//#include "bootstrap.h"
#include <boost/filesystem.hpp>

const int DEG = 180;
const double WGS84_A = 6378137.0;
const double WGS84_B = 6356752.314245;
const double EARTH_RADIUS = 6371e3;

double toRadian(double deg);

std::string parseFileNameFromPath(std::string path);

/*
double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
  double lat1r, lon1r, lat2r, lon2r, u, v;
  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  u = sin((lat2r - lat1r)/2);
  v = sin((lon2r - lon1r)/2);
  return 2.0 * EARTH_RADIUS * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}
*/
struct Location
{
    double longitude;
    double latitude;
    double altitude;
    double dop;
    bool isEmpty = true;

    double distanceTo(Location loc)
    {
        auto b = 2;
        auto longRad = toRadian(this->longitude);
        auto latRad = toRadian(this->latitude);

        auto locLongRad = toRadian(loc.longitude);
        auto locLatRad = toRadian(loc.latitude);

        auto u = sin((locLatRad - latRad) / b);
        auto v = sin((locLongRad - longRad) / b);
        return 2.0 * EARTH_RADIUS * asin(sqrt(u * u + cos(latRad) * cos(locLatRad) * v * v));
    }

    /**
     * Uses WGS84 model for GPS distance. See https://github.com/mapillary/OpenSfM/blob/master/opensfm/geo.py
     */
    double wgDistanceTo(Location loc)
    {
        auto p1 = this->ecef();
        auto p2 = loc.ecef();

        auto dist = sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2) + pow((p1.z - p2.z), 2));
        return dist;
    }

    /**
     * CHeck results for ecef function here http://
     * www.oc.nps.edu/oc2902w/coord/llhxyz.htm
     */
    cv::Point3d ecef()
    {
        auto b = 2.0;
        auto a2 = pow(WGS84_A, b);
        auto b2 = pow(WGS84_B, b);
        auto longRad = toRadian(longitude);
        auto latRad = toRadian(latitude);
        auto l = 1.0 / sqrt(a2 * pow(cos(latRad), b) + b2 * pow(sin(latRad), 2));
        auto x = (a2 * l + this->altitude) * cos(latRad) * cos(longRad);
        auto y = (a2 * l + this->altitude) * cos(latRad) * sin(longRad);
        auto z = (b2 * l + this->altitude) * sin(latRad);

        return cv::Point3d(x, y, z);
    }

    cv::Point3d getTopcentricLocationCoordinates(std::map<std::string, double> reference) {
        const cv::Mat t = Location::topcentricTransformFromReferenceLLA(reference).inv();
        const auto locEcef = ecef();
        const auto tx = t.at<double>(0, 0) * locEcef.x + t.at<double>(0, 1) * locEcef.y + t.at<double>(0, 2) * locEcef.z
            + t.at<double>(0, 3);
        const auto ty = t.at<double>(1, 0) * locEcef.x + t.at<double>(1, 1) * locEcef.y + t.at<double>(1, 2) * locEcef.z
            + t.at<double>(1, 3);
        const auto tz = t.at<double>(2, 0) * locEcef.x + t.at<double>(2, 1) * locEcef.y + t.at<double>(2, 2) * locEcef.z
            + t.at<double>(2, 3);

        return { tx,ty, tz };
    }

    friend std::ostream &operator<<(std::ostream &os, const Location &loc)
    {
        os << loc.latitude << " " << loc.longitude << " " << loc.altitude;
        return os;
    }

    static cv::Mat topcentricTransformFromReferenceLLA(std::map<std::string, double> referenceLLA) {
        const auto lat = referenceLLA["lat"];
        const auto lon = referenceLLA["lon"];
        const auto alt = referenceLLA["alt"];

        Location refLocation{ lon, lat, alt, 0.0 };
        const auto refLlaEcef = refLocation.ecef();
        const auto sa = sin(toRadian(lat));
        const auto ca = cos(toRadian(lat));
        const auto so = sin(toRadian(lon));
        const auto co = cos(toRadian(lon));

        cv::Matx<double, 4, 4> topCentricReference{
            -so, -sa * co, ca * co, refLlaEcef.x,
            co, -sa * so, ca * so, refLlaEcef.y,
            0, ca, sa, refLlaEcef.z,
            0, 0, 0, 1
        };

        return cv::Mat(topCentricReference);
    }
};

struct ImageMetadata
{
    Location location;
    int height;
    int width;
    std::string projectionType;
    std::string cameraMake;
    std::string cameraModel;
    int orientation;
    double captureTime;
};

class Img
{
public:
    using CameraMake = std::string;
    using CameraModel = std::string;
    using CameraMakeAndModel = std::tuple<CameraMake, CameraModel>;

private:
    std::string imageFileName;
    ImageMetadata metadata;
    static Location _extractCoordinatesFromExif(Exiv2::ExifData exifData);
    static double _extractPhysicalFocalFromExif(Exiv2::ExifData exifData);
    // TODO Implement unimplemented functions in image class
    std::string _extractProjectionTypeFromExif(Exiv2::ExifData exifData);
    static CameraMakeAndModel _extractMakeAndModelFromExif(Exiv2::ExifData exifData);
    static double _extractDopFromExif(Exiv2::ExifData imageExifData);
    static int _extractOrientationFromExif(Exiv2::ExifData imageExifData);

public:
    Img() : imageFileName(), metadata() {};
    //Constructs an img class given the path to the image 
    Img(std::string imageFileName);
    Img(std::string fileName, ImageMetadata metadata) : imageFileName(fileName), metadata(metadata) {};
    const ImageMetadata& getMetadata() const;
    const std::string& getFileName() const;
    ImageMetadata& getMetadata();
    static ImageMetadata extractExifFromImage(std::string imagePath);
    static void extractExifFromFile(std::string imageExifFile, ImageMetadata& metadata);

};

