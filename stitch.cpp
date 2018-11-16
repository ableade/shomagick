#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <utility>
#include "exiv2/exiv2.hpp"
//#include <pcl/point_cloud.h>
//#include <pcl/kdtree/kdtree_flann.h>
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
#include "shomatcher.hpp"
#include "shotracking.h"

using namespace boost::filesystem;
using cv::BFMatcher;
using cv::DMatch;
using cv::imread;
using std::cout;
using std::endl;
using std::function;
using std::pair;
using std::set;
using std::setprecision;
using std::string;
using std::vector;
using namespace cv::xfeatures2d;

bool pairCompare(pair<string, double> a, pair<string, double> b)
{
  return a.second < b.second;
}

int main(int argc, char *argv[])
{
  std::ofstream outfile;
  std::ofstream harvFile;
  std::ofstream wsgFile;
  if (argc < 2)
  {
    cout << "Program usage: <flight session directory>" << endl;
    exit(1);
  }
  FlightSession flight(argv[1]);
  ShoMatcher shoMatcher(flight);
  
  shoMatcher.getCandidateMatches();
  shoMatcher.runRobustFeatureDetection();

  /*
harvFile.open("harv.csv"); wsgFile.open("wsg.csv");
//Compute nearest neighbors using haversine formula
harvFile << "Image name, Neighboring image name, Distance,"<<endl;
wsgFile << "Image name, Neighboring image name, Distance,"<<endl;
for(auto i =0; i< v.size();++i) {
 vector<pair<string,double > > harvDistances;
 vector<pair<string,double > > wsgDistances;
 auto currentImage = parseFileNameFromPath(v[i].path().string());
 cout<<std::fixed<<std::setprecision(7);
 cout<<"Current image is " << currentImage<<endl;
 cout << "Coordinates for current image are "<<mosaicImages[i].location<<endl;
 cout << "ECEF for current image is " << mosaicImages[i].location.ecef()<<endl;
 for (int j=0; j < v.size();++j) {
   if (j == i) {
     continue;
   }
           //calculate distance from v[i] tp v[j]

   auto harvesineDistance = mosaicImages[i].location.distanceTo(mosaicImages[j].location);
   auto wsgDistance = mosaicImages[i].location.wgDistanceTo(mosaicImages[j].location);

   auto harvPair = make_pair(mosaicImages[j].fileName, harvesineDistance);
   auto wsgPair = make_pair(mosaicImages[j].fileName, wsgDistance);
   harvDistances.push_back(harvPair);
   wsgDistances.push_back(wsgPair);
           //cout << "Coordinates for compare image are "<<mosaicImages[j].location<<endl;
           //cout << "Haversine distance is "<< harvesineDistance<<endl;
           //cout << "ECEF for compare image is "<<mosaicImages[j].location.ecef()<<endl;
           //cout << "Wsg distance is "<<wsgDistance<<endl;
 }
 sort (harvDistances.begin(), harvDistances.end(), pairCompare);
 sort (wsgDistances.begin(), wsgDistances.end(), pairCompare);
 for(int j=0; j<k; ++j) {
   harvFile << currentImage<<","<<harvDistances[j].first<<","<<std::fixed<<std::setprecision(7)<<harvDistances[j].second<<","<<endl;
   wsgFile  << currentImage <<"," << wsgDistances[j].first<<","<<std::fixed<<std::setprecision(7)<<wsgDistances[j].second<<","<<endl;
 }
 cout << "*********************"<<endl;
}
harvFile.close();
wsgFile.close();
*/
}
