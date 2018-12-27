#include <iostream>
#include <string>
#include <fstream>
#include <utility>
#include <map>
#include <iomanip>
#include <array>
#include <algorithm>
#include <vector>

#include "csv.h"

using std::cout;
using std::endl;
using std::find;
using std::ifstream;
using std::map;
using std::ofstream;
using std::pair;
using std::string;
using std::vector;

struct Matchdetails
{
	double matchPercentage;
	int numMatches;
};

int main_(int argc, char *argv[])
{
	if (argc < 3)
	{
		cout << "Program usage: <Keypoints file> <Distances file>" << endl;
		exit(1);
	}
	io::CSVReader<4> in(argv[1]);
	io::CSVReader<3> nearest(argv[2]);
	ofstream outfile("datajoin.csv");
	vector<std::array<char, 100>> header;
	outfile << "Query Image Name, Match Image Name, Num Matches, Keypoint Match Percentage, Distance," << endl;

	int numMatches;
	string queryImageName, matchImageName;
	double matchKeypointPercentage, distance;
	map<pair<string, string>, Matchdetails> imagePairs;

	in.read_header(io::ignore_extra_column, "Query image name", "Match image name", "num_matches", "Keypoint Match Percentage");
	while (in.read_row(queryImageName, matchImageName, numMatches, matchKeypointPercentage))
	{
		cout << "Making a read" << endl;
		auto aPair = make_pair(queryImageName, matchImageName);
		Matchdetails detail{matchKeypointPercentage, numMatches};
		imagePairs[aPair] = detail;
	}
	//Finished parsing keypoints file, parse distances file.
	nearest.read_header(io::ignore_extra_column, "Image name", "Neighboring image name", "Distance");
	while (nearest.read_row(queryImageName, matchImageName, distance))
	{
		auto aPair = make_pair(queryImageName, matchImageName);
		if (imagePairs.find(aPair) != imagePairs.end())
		{
			outfile << queryImageName << "," << matchImageName << "," << imagePairs[aPair].numMatches << ","
					<< imagePairs[aPair].matchPercentage << "," << std::fixed << std::setprecision(10) << distance << "," << endl;
		}
	}
	outfile.close();

	return 0;
}