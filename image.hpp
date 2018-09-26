#ifndef SHOIMAGE_HPP_
#define SHOIMAGE_HPP_

#include <string>
#include <math.h>

float toRadian (float deg) {
	return deg * M_PI/180;
}

struct Location {
	float longitude;
	float latitude;

	float distanceTo(Location loc) {
		auto earthRadius = 6371e3;
		auto longRad = toRadian(this.longitude);
		auto latRad = toRadian(this.latitude);

		auto locLongRad = toRadian(loc.longitude);
		auto locLatRad = toRadian(loc.latitude);

		auto dLat = locLatRad - latRad;
		auto dLong = locLongRad - longRad;

		auto  a = sin(dLat/2) * sin(dLat/2) + cos(longRad) * cos(locLongRad) * sin(dLong/2) * sin(dLong/2);
		auto c =2 * atan(sqrt(a), sqrt(1-a));
		return earthRadius * c;
	}
};


struct Img {
    std::string fileName;
    Location location;
};

#endif
