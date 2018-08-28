#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <set>

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

int getImages(string path, vector<Img>& imgs, vector<Mat>& descps) {
	auto count =0;
	SIFTDetector sift;
	vector<directory_entry> v;
    assert (is_directory(path));
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
    for (entry : v) {
    	vector<KeyPoint> keypoints;
    	Img img;
    	Mat imge = cv::imread(entry.path().string());
        if (imge.empty())
            continue;
        count++;
        Mat descriptors;
    	img.fileName = entry.path().string();
    	sift(imge, Mat(), keypoints, descriptors);
    	imgs.push_back(img);
    	descps.push_back(descriptors);
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