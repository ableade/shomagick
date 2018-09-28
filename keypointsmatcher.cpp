#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <set>
#include <map>
#include <boost/dynamic_bitset.hpp>
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
using std::set;
using std::pair;
using std::map;
using std::make_pair;
using cv::FlannBasedMatcher;
using namespace cv::xfeatures2d;

vector<directory_entry> getImages(string path, vector<Mat>& descps) {
	auto count =0;
	Detector <SURF> myDetector;
	vector<directory_entry> v;
    assert (is_directory(path));
#if 0
    copy(directory_iterator(path), directory_iterator(), back_inserter(v));
#else
    copy_if(
    	directory_iterator(path),
    	directory_iterator(),
    	back_inserter(v),
    	[]( const directory_entry& e ){
    		return is_regular_file( e );
    	}
    );
#endif
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
	std::ofstream outFile;
	string outputFileName = "results.csv";
	outFile.open(outputFileName);

	auto v = getImages(imageDirectory.string(), trainDescriptors);
	cout << "Number of descriptors is "<< trainDescriptors.size()<<endl;
	Matcher<FlannBasedMatcher> matcher(trainDescriptors);
	outFile << "Query image name, Match image name, num_matches, Keypoint Match Percentage, "<<endl;

	for(int i=0; i< v.size(); i++) {
		set< pair <string, string> > rawMatches;
		map< pair <string, string>, int> matchCount;
		vector<   boost::dynamic_bitset<> > percentageMatches(v.size(), boost::dynamic_bitset<>(trainDescriptors[i].rows));
		assert (percentageMatches.size()== v.size());
		assert(percentageMatches.front().size() == trainDescriptors[i].rows);
		cout << "Now finding matches for image: "<<v[i].path().string()<<endl;
		vector< vector <DMatch> > matches;
		matcher.match(trainDescriptors[i], matches, 3);
		for(auto matchList: matches) {
			for (auto match: matchList) {
				auto queryFileName = parseFileNameFromPath(v[i].path().string());
				auto trainFileName = parseFileNameFromPath(v[match.imgIdx].path().string());
				if (match.imgIdx != i) {
					auto aPair = make_pair(queryFileName, trainFileName);
					auto search = rawMatches.find(aPair);
					if (search != rawMatches.end()) {
						matchCount[aPair] ++;
					} else {
						rawMatches.insert(aPair);
						matchCount[aPair] = 1;
					}
					percentageMatches[match.imgIdx][match.queryIdx] = 1;
				}
			}
			cout <<"Finished examining matchlist"<<endl;
			for(int j =0; j< percentageMatches.size(); j++) {
				auto queryFileName = parseFileNameFromPath(v[i].path().string());
				auto trainFileName = parseFileNameFromPath(v[j].path().string());
				if (i == j)
					continue;
				auto pct = percentageMatches[j].count() / float(trainDescriptors[i].rows);
				outFile<< queryFileName << "," << trainFileName << ","<< matchCount[make_pair(queryFileName, trainFileName)]
				<<"," << pct << ","<<endl;
			}
		}
		cout << "Examined all matches "<<endl;	
	}
	outFile.close();
}