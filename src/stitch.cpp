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
#include "reconstructor.h"

using namespace boost::filesystem;
using cv::BFMatcher;
using cv::DescriptorExtractor;
using cv::DMatch;
using cv::FeatureDetector;
using cv::imread;
using cv::ORB;
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
  string cameraCalibrationFile;
  if (argc < 2)
  {
    cout << "Program usage: <flight session directory> optional -- <camera calibration file>" << endl;
    exit(1);
  }
  FlightSession flight;
  argc > 2 ? flight = FlightSession(argv[1], argv[2]) : flight = FlightSession(argv[1]);

  ShoMatcher shoMatcher(flight);

  //**** Begin Matching Pipeline ******
  if (argc > 3)
  {
    //A candidate file was provided
    const auto candidateFile = argv[3];
    cout << "Using candidate file " << candidateFile << std::endl;
    shoMatcher.getCandidateMatchesFromFile(candidateFile);
  }
  else
  {
    shoMatcher.getCandidateMatchesUsingSpatialSearch();
  }
  shoMatcher.extractFeatures();
  shoMatcher.runRobustFeatureMatching();
  //******End matching pipeline******

  //***Begin tracking pipeline *****
  ShoTracker tracker(flight, shoMatcher.getCandidateImages());
  vector<pair<ImageFeatureNode, ImageFeatureNode>> featureNodes;
  vector<FeatureProperty> featureProps;
  tracker.createFeatureNodes(featureNodes, featureProps);
  tracker.createTracks(featureNodes);
  auto tracksGraph = tracker.buildTracksGraph(featureProps);
  cout << "Created tracks graph " << endl;
  cout << "Number of vertices is " << tracksGraph.m_vertices.size() << endl;
  cout << "Number of edges is " << tracksGraph.m_edges.size() << endl;
  auto commonTracks = tracker.commonTracks(tracksGraph);
  Reconstructor reconstructor(flight, tracksGraph, tracker.getTrackNodes(), tracker.getImageNodes());

  string image1 = "0060_SONY.jpg";
  string image2 = "0062_SONY.jpg";

  auto imageNodes = tracker.getImageNodes();

  auto im1 = imageNodes[image1];
  auto im2 = imageNodes[image2];

  TwoViewPose t;
  cv::Mat mask;
  for (const auto &track : commonTracks)
  {
    if (track.imagePair.first == image1 && track.imagePair.second == image2) 
    {
      cout << "Number of common tracks is " << track.commonTracks.size() << endl;
      t = reconstructor.recoverTwoCameraViewPose(im1, im2, track.commonTracks, mask);
      cout << "Essential matrix: \n" << get<0>(t) << "*******\n";
      auto rotation = get<1>(t);
      cv::Rodrigues(rotation, rotation);
      cout << "Rotation: \n " << rotation << "*******\n";
      cout << " Translation: \n"<<get<2>(t)<<endl;
      cv::Mat hom = reconstructor.computePlaneHomography(track);
      cout << "The homography found was " << hom << endl;


    vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
    int solutions = decomposeHomographyMat(hom, flight.getCamera().getNormalizedKMatrix(), Rs_decomp, ts_decomp, normals_decomp);
    cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
    for (int i = 0; i < solutions; i++)
    {
      cv::Mat rvec_decomp;
      Rodrigues(Rs_decomp[i], rvec_decomp);
      cout << "Solution " << i << ":" << endl;
      cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
      //cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
      cout << "tvec from homography decomposition: " << ts_decomp.at(i).t() <<endl;
      //cout << "tvec from camera displacement: " << t_1to2.t() << endl;
      cout << "plane normal from homography decomposition: " << normals_decomp.at(i).t() << endl;
    }
    }
  }
  //reconstructor.runIncrementalReconstruction(tracker);
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
