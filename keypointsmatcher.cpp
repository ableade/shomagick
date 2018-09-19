#include <iostream>
#include <string>
#include "detector.h"
#include "image.hpp"
#include <boost/filesystem.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace boost::filesystem;
using cv::Mat;
using std::string;
using std::endl;
using std::vector;
using cv::FlannBasedMatcher;
using namespace cv::xfeatures2d;

vector<directory_entry> getImages(string path, vector<Mat>& descps) {
	auto count =0;
	Detector <SURF> myDetector;
	vector<directory_entry> v;
    assert (is_directory(path));
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
    for (auto entry : v) {
    	vector<KeyPoint> keypoints;
    	Mat imge = cv::imread(entry.path().string()); 
        if (imge.empty())
            continue;
        count++;
        Mat descriptors;
    	myDetector(imge, Mat(), keypoints, descriptors);
    	descps.push_back(descriptors);
    }
    return v;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Program usage: <images directory>"<<endl;
		exit(1);
	}
	vector<Mat> trainDescriptors;
	path imageDirectory(argv[1]);

	auto v = getImages(imageDirectory.string(), trainDescriptors);
	Matcher<FlannBasedMatcher> matcher(trainDescriptors);

	for(int i=0; i< v.size(); i++) {
		cout << "Now finding matches for image: "<<v[i].path().string()<<endl;
		vector< vector <DMatch> > matches;
		matcher.match(trainDescriptors[i], matches, 3);
		for(auto matchList: matches) {
			for (auto match: matchList) {
				if (match.imgIdx != i) {
					cout << "Image match was at "<<match.imgIdx<<endl;
					cout<< "Match with image "<<v[match.imgIdx].path().string()<<endl;
				}
			}
			cout <<"Finished examining matchlist"<<endl;
		}
		cout << "Examined all matches "<<endl;
		cout << "****************************"<<endl;
	}
}