#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <set>
#include "exiv2/exiv2.hpp"

#include "sift.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace boost::filesystem;
using std::vector;
using std::set;
using cv::DMatch;
using cv::BFMatcher;
using std::cout;
using std::endl;
using std::string;
using namespace cv::xfeatures2d;

struct Img {
    string fileName;
};

struct Location {
	float longitude;
	float latitude;
};

Location getCoodinates(string path) {
    Location loc;
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(path);
    assert(image.get() != 0);
    image->readMetadata();
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty()) {
    	std::string error(path);
    	error += ": No Exif data found in the file";
    	throw Exiv2::Error(1, error);
    }
    Exiv2::ExifData::const_iterator end = exifData.end();
    Exiv2::Value::AutoPtr latV = Exiv2::Value::create(Exiv2::signedRational);
    Exiv2::Value::AutoPtr longV = Exiv2::Value::create(Exiv2::signedRational);
    auto longitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLongitude");
    auto latitudeKey = Exiv2::ExifKey("Exif.GPSInfo.GPSLatitude");
    auto latPos = exifData.findKey(latitudeKey);
    auto longPos = exifData.findKey(longitudeKey);
    if (latPos == exifData.end() || longPos == exifData.end()) throw Exiv2::Error(1, "Key not found");
    // Get a pointer to a copy of the value
    latV = latPos->getValue();
    longV = longPos->getValue();
    auto latitude = latV->toFloat() + (latV->toFloat(1) / 60.0) + (latV->toFloat(2) / 3600.0); 
    auto longitude =longV->toFloat() + (longV->toFloat(1) / 60.0) + (longV->toFloat(2) / 3600.0);
										
    cout << "Latitude of this image is "<<latitude<<endl;
    cout << "Longitude of this image is "<<longitude<<endl;
    for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i) {
        const char* tn = i->typeName();
        std::cout << std::setw(44) << std::setfill(' ') << std::left
                  << i->key() << " "
                  << "0x" << std::setw(4) << std::setfill('0') << std::right
                  << std::hex << i->tag() << " "
                  << std::setw(9) << std::setfill(' ') << std::left
                  << (tn ? tn : "Unknown") << " "
                  << std::dec << std::setw(3)
                  << std::setfill(' ') << std::right
                  << i->count() << "  "
                  << std::dec << i->value()
                  << "\n";
    }
    return loc;
}

int getImages(string path, vector<Img>& imgs, vector<Mat>& descps) {
	auto count =0;
	SIFTDetector sift;
	vector<directory_entry> v;
    assert (is_directory(path));
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
    for (auto entry : v) {
    	vector<KeyPoint> keypoints;
    	Img img;
    	Mat imge = cv::imread(entry.path().string());
        if (imge.empty())
            continue;
        count++;
        getCoodinates(entry.path().string());
        Mat descriptors;
    	img.fileName = entry.path().string();
    	//sift(imge, Mat(), keypoints, descriptors);
    	imgs.push_back(img);
    	//descps.push_back(descriptors);
    }
    cout << "Found "<< count << "usable images"<<endl;
    return count;
}


int main(int argc, char* argv[]) {
	vector<Mat> trainDescriptors;
	vector<Img> mosaicImages;
	if (argc < 2) {
		cout << "Program usage: <images directory>"<<endl;
		exit(1);
	}
	path imageDirectory(argv[1]);
	int imageCount = getImages(imageDirectory.string(), mosaicImages, trainDescriptors);
	SIFTMatcher<BFMatcher> matcher(trainDescriptors);
	//computing matches among all images
	for(int i=0; i< imageCount; i++) {
		vector< vector <DMatch> > matches;
		matcher.match(trainDescriptors[i], matches, 3);
		for(auto matchList: matches) {
			for (auto match: matchList) {
				cout << "Image match was at "<<match.imgIdx<<endl;
			}
		}
	}
}
