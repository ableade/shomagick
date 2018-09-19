#ifndef SHOIMAGE_HPP_
#define SHOIMAGE_HPP_

#include <string>

struct Location {
	float longitude;
	float latitude;
};


struct Img {
    std::string fileName;
    Location location;
};

#endif
