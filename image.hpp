#ifndef IMAGE_HPP_
#define IMAGE_HPP_

struct Location {
	float longitude;
	float latitude;
};


struct Img {
    string fileName;
    Location location;
};

#endif
