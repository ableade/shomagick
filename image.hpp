#ifndef SHOIMAGE_HPP_
#define SHOIMAGE_HPP_

#include <string>
#include <cmath>

const int DEG = 180;

float toRadian (float deg) {
	return deg * M_PI/DEG;
}

struct Location {
	float longitude;
	float latitude;

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
};


struct Img {
    std::string fileName;
    Location location;
};

#endif
