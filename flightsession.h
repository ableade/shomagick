#ifndef SHOFLIGHTSESSION_HPP_
#define SHOFLIGHTSESSION_HPP_
#include <string>
#include <vector>
#include "image.hpp"
#include <boost/filesystem.hpp>

using std::string;
using std::vector;
using namespace boost::filesystem;

class FlightSession {

private:
	vector<Img> imageData;
	string imageDirectory;
	path imageDirectoryPath;
	path imageFeaturesPath;

public:
	FlightSession(string imageDirectory);
	Location getCoordinates(string imagePath);
	const vector<Img> getImageSet();
	const path getImageDirectoryPath() const;
	const path getImageFeaturesPath() const;
};
#endif