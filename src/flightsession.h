#ifndef SHOFLIGHTSESSION_HPP_
#define SHOFLIGHTSESSION_HPP_
#include <string>
#include <vector>
#include "image.hpp"
#include <map>
#include <boost/filesystem.hpp>

typedef std::pair< std::vector<cv::KeyPoint> , cv::Mat> ImageFeatures;

class FlightSession {

private:
	std::vector<Img> imageData;
	std::string imageDirectory;
	boost::filesystem::path imageDirectoryPath;
	boost::filesystem::path imageFeaturesPath;
	boost::filesystem::path imageMatchesPath;
	boost::filesystem::path imageTracksPath;

public:
	FlightSession(string imageDirectory);
	Location getCoordinates(string imagePath);
	std::vector<Img> getImageSet() const;
	const boost::filesystem::path getImageDirectoryPath() const;
	const boost::filesystem::path getImageFeaturesPath() const;
	const boost::filesystem::path getImageMatchesPath() const;
	const boost::filesystem::path getImageTracksPath() const;
	bool saveTracksFile(std::map <int, std::vector <int>> tracks);
	int getImageIndex(string imageName) const;
	std::map <string, std::vector<cv::DMatch>> loadMatches(string fileName);
	bool saveImageFeaturesFile(string imageName, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat descriptors);
	bool saveMatches(string fileName, std::map<string ,std::vector<cv::DMatch>> matches);
	ImageFeatures loadFeatures(string imageName);

};
#endif