#include <iostream>
#include <string>
#include <fstream>
#include <array>
#include <vector>

using std::cout;
using std::string;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::endl;

int main (int argc, char* argv[]) {
	if (argc < 3) {
		cout << "Program usage: <results file> <distances file>"<<endl;
		exit(1);
	}

	ifstream infile(argv[1]);
	ofstream outfile("datajoin.csv");
	vector<std::array<char, 100>> header;
	outfile << "Query Image Name, Match Image Name, Num Matches, Keypoint Match Percentage, Distance,"<<endl;
	string h;
	getline(infile, h);

	return 0;
}