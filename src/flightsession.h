#ifndef SHOFLIGHTSESSION_HPP_
#define SHOFLIGHTSESSION_HPP_
#include <string>
#include <vector>
#include "image.hpp"
#include "camera.h"
#include <map>
#include <boost/filesystem.hpp>

class ImageFeatures {
	public:
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		std::vector<cv::Scalar> colors;
};

class FlightSession
{

  private:
	std::vector<Img> imageData;
	std::string imageDirectory;
	boost::filesystem::path imageDirectoryPath;
	boost::filesystem::path imageFeaturesPath;
	boost::filesystem::path imageMatchesPath;
	boost::filesystem::path imageTracksPath;
	Camera camera;

  public:
	FlightSession();
	FlightSession(string imageDirectory, string calibFile=string());
	Location getCoordinates(string imagePath);
	std::vector<Img> getImageSet() const;
	const boost::filesystem::path getImageDirectoryPath() const;
	const boost::filesystem::path getImageFeaturesPath() const;
	const boost::filesystem::path getImageMatchesPath() const;
	const boost::filesystem::path getImageTracksPath() const;
	bool saveTracksFile(std::map<int, std::vector<int>> tracks);
	int getImageIndex(string imageName) const;
	std::map<string, std::vector<cv::DMatch>> loadMatches(string fileName);
	bool saveImageFeaturesFile(string imageName, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat& descriptors, 
	const std::vector<cv::Scalar>& colors);
	bool saveMatches(string fileName, const std::map<string, std::vector<cv::DMatch>>& matches);
	ImageFeatures loadFeatures(string imageName);
	Camera getCameraFromCalibrationFile(string);
	void initializeCameraFromExifMetaData();
	const Camera& getCamera() const;
};
#endif