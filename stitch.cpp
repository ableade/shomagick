#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <set>
#include "exiv2/exiv2.hpp"
//#include <pcl/point_cloud.h>
//#include <pcl/kdtree/kdtree_flann.h>

#include "detector.h"
#include "image.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "kdtree.h"

using namespace boost::filesystem;
using std::vector;
using std::set;
using cv::DMatch;
using cv::BFMatcher;
using std::cout;
using std::endl;
using std::string;
using std::function;
using std::setprecision;
using namespace cv::xfeatures2d;

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

    loc.longitude = longitude;
    loc.latitude = latitude;

    cout << "Latitude: "<< setprecision(10)<<latitude << "longitude: "<<longitude <<endl;
    return loc;
}

vector<directory_entry> getImages(string path, vector<Img>& imgs) {
	auto count =0;
	Detector <SURF> myDetector;
	vector<directory_entry> v;
    assert (is_directory(path));
     copy_if(
    	directory_iterator(path),
    	directory_iterator(),
    	back_inserter(v),
    	[]( const directory_entry& e ){
    		return is_regular_file( e );
    	}
    );
    for (auto entry : v) {
    	vector<KeyPoint> keypoints;
    	Img img;
    	img.fileName = parseFileNameFromPath(entry.path().string());
        cout << "Parsing GPS information for this image"<< entry.path().string()<<endl;
        auto loc = getCoodinates(entry.path().string());
        img.location = loc;
        imgs.push_back(img);
    }
    cout << "Found "<< count << "usable images"<<endl;
    return v;
}


int main(int argc, char* argv[]) {
    auto dimensionality =2;
	vector<Img> mosaicImages;
	if (argc < 2) {
		cout << "Program usage: <images directory>"<<endl;
		exit(1);
	}
	path imageDirectory(argv[1]);
    auto v = getImages(imageDirectory.string(), mosaicImages);

    //Iniitalize K-D tree for GPS location data
    void *kd;
    kd = kd_create(dimensionality);

    for (auto& img : mosaicImages) {
        double pos[dimensionality] = {img.location.longitude, img.location.latitude};
        void *dt = &img;
        assert(kd_insert(static_cast<kdtree*>(kd), pos, dt) == 0); 
    }

    //Compute the nearest neighbors for every image
    for (auto i = 0; i< v.size(); i++) {
        cout << "Image: "<< v[i].path().string()<<endl;
        auto range = 0.00010;
        void *result_set;

        double pt[] = {mosaicImages[i].location.longitude, mosaicImages[i].location.latitude}; result_set = kd_nearest_range(static_cast<kdtree*>(kd), pt, range);
        cout << "Size of the result set is "<<kd_res_size(static_cast<kdres*>(result_set))<<endl;
        
        cout << "Closest neighbors for this image are "<<endl;
        while(!kd_res_end(static_cast<kdres*>(result_set))) {
        	auto current = kd_res_item(static_cast<kdres*>(result_set), pt);
        	if (current == nullptr) {
        		cout << "Got NULL pointer"<<endl;
        		continue;
        	}
        	auto img = static_cast<Img*>(current); 
        	cout << "Image name was "<<img->fileName << endl;
        	kd_res_next(static_cast<kdres*>(result_set));
        }
         cout <<endl;
    }

}
