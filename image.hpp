#ifndef SHOIMAGE_HPP_
#define SHOIMAGE_HPP_

#include <string>
#include <cmath>

using std::string;

const int DEG = 180;
const float WGS84_A = 6378137.0;
const float WGS84_B = 6356752.314245;


float toRadian (float deg) {
	return deg * M_PI/DEG;
}

string parseFileNameFromPath(string path) {
	int index = path.find_last_of('/');
	return path.substr(index+1);
}

struct Location {
	float longitude;
	float latitude;
	float altitude;

	float distanceTo(Location loc) {
		auto b =2;
		auto earthRadius = 6371e3;
		auto longRad = toRadian(this->longitude);
		auto latRad = toRadian(this->latitude);

		auto locLongRad = toRadian(loc.longitude);
		auto locLatRad = toRadian(loc.latitude);

		auto dLat = locLatRad - latRad;
		auto dLong = locLongRad - longRad;
		
		auto  a = sin(dLat/b) * sin(dLat/b) + cos(longRad) * cos(locLongRad) * sin(dLong/b) * sin(dLong/b);
		auto c =b * atan2(sqrt(a), sqrt(1-a));
		return earthRadius * c;
	}

	/**
	 * Uses WGS84 model for GPS distance. See https://github.com/mapillary/OpenSfM/blob/master/opensfm/geo.py
	 */
	float distanceTo(float longitude, float latitude, float altitude) {

	}

	float ecef() {
		
	}

};


struct Img {
    std::string fileName;
    Location location;
};

#endif
