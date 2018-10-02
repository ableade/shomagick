#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <map>
#include <array>
#include <algorithm>
#include <vector>

#include "csv.h"

using std::cout;
using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::pair;
using std::find;
using std::endl;
using std::map;

struct Matchdetails {
	double matchPercentage;
	int numMatches;
};

int main (int argc, char* argv[]) {
	if (argc < 3) {
		cout << "Program usage: <Keypoints file> <Distances file>"<<endl;
		exit(1);
	}

	ifstream infile(argv[1]);
	ofstream outfile("datajoin.csv");
	vector<std::array<char, 100>> header;
	outfile << "Query Image Name, Match Image Name, Num Matches, Keypoint Match Percentage, Distance,"<<endl;
	string h;
	getline(infile, h);

	int numMatches;
	string queryImageName, matchImageName;
	double matchKeypointPercentage, distance;
	map <pair<string, string>, Matchdetails> imagePairs;

	while(infile >> queryImageName >> matchImageName >> numMatches >> matchKeypointPercentage) {
		cout << "Making a read"<<endl;
		auto aPair = make_pair(queryImageName, matchImageName);
		Matchdetails detail{matchKeypointPercentage, numMatches};
		imagePairs[aPair] = detail;

	}
	infile.close();

	//Finished parsing keypoints file, parse distances file.
	infile.open(argv[2]);
	getline(infile, h);

	while (infile >> queryImageName >> matchImageName >> distance) {
		auto aPair = make_pair(queryImageName, matchImageName);
		if (imagePairs.find(aPair) != imagePairs.end()) {
			outfile << queryImageName << ","<<matchImageName << "," << imagePairs[aPair].numMatches << ","
			<<imagePairs[aPair].matchPercentage<< ","<<distance<<","<<endl;
		}
	}
	infile.close(); outfile.close();
	
	return 0;
}