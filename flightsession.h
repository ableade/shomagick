#ifndef SHOFLIGHTSESSION_HPP_
#define SHOFLIGHTSESSION_HPP_
#include <string>
#include <vector>
#include "image.hpp"
#include <boost/filesystem.hpp>

class FlightSession {

private:
	std::vector<Img> imageData;
	std::string imageDirectory;
	boost::filesystem::path imageDirectoryPath;
	boost::filesystem::path imageFeaturesPath;
	boost::filesystem::path imageMatchesPath;

public:
	FlightSession(string imageDirectory);
	Location getCoordinates(string imagePath);
	std::vector<Img> getImageSet() const;
	const boost::filesystem::path getImageDirectoryPath() const;
	const boost::filesystem::path getImageFeaturesPath() const;
	const boost::filesystem::path getImageMatchesPath() const;
	int getImageIndex(string imageName) const;

};
#endif