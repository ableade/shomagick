#pragma once

#include <string>
#include <cmath>
#include <fstream>
#include <opencv2/core.hpp>
#include "bootstrap.h"
#include <boost/filesystem.hpp>

const int DEG = 180;
const float WGS84_A = 6378137.0;
const float WGS84_B = 6356752.314245;
const float EARTH_RADIUS = 6371e3;

inline double toRadian(double deg)
{
	return deg * M_PI / DEG;
}

inline std::string parseFileNameFromPath(std::string path)
{
	return boost::filesystem::path{path}.filename().string();
}


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

	float distanceTo(Location loc)
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
	float wgDistanceTo(Location loc)
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
		auto longRad = toRadian(this->longitude);
		auto latRad = toRadian(this->latitude);

		auto l = 1.0 / sqrt(a2 * pow(cos(latRad), b) + b2 * pow(sin(latRad), 2));
		auto x = (a2 * l + this->altitude) * cos(latRad) * cos(longRad);
		auto y = (a2 * l + this->altitude) * cos(latRad) * sin(longRad);
		auto z = (b2 * l + this->altitude) * sin(latRad);

		return cv::Point3d(x, y, z);
	}

	friend std::ostream &operator<<(std::ostream &os, const Location &loc)
	{
		os << loc.latitude << " " << loc.longitude << " " << loc.altitude;
		return os;
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
    std::string orientation;
};

struct Img
{
	std::string fileName;
    ImageMetadata metadata;

	Img() : fileName(), metadata() {};
	Img(std::string fileName, ImageMetadata metadata) : fileName(fileName) ,  metadata(metadata) {};
};

