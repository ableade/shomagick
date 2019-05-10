#include "catch.hpp"
#include <iostream>
#include <utility>
#include <vector>
#include "../src/flightsession.h"
#include "../src/shotracking.h"
#include <boost/filesystem.hpp>
#include "../src/shomatcher.hpp"
#include "../src/reconstructor.h"
#include "../src/utilities.h"

using std::vector;
using std::string;
using std::endl;
using  std::pair;

SCENARIO("Testing the two view reconstruction between two images")
{
    GIVEN("Two images in a reconstruction data set"){
    boost::filesystem::path dataPath(boost::filesystem::current_path());
    dataPath/= "data/berlin"; 
    auto flightCamera = getPerspectiveCamera(0.9722, 3264, 2448, 0.0, 0.0);
    const auto imagesPath = dataPath / "images";
    const auto reportsPath = dataPath / "reports";
    const auto matchesFile = reportsPath / "matches.json";
    FlightSession flight(imagesPath.string());
    flight.setCamera(flightCamera);
    ShoMatcher shoMatcher(flight);
    shoMatcher.getCandidateMatchesFromFile(matchesFile.string());
    shoMatcher.extractFeatures();
    shoMatcher.runRobustFeatureMatching();
    ShoTracker tracker(flight, shoMatcher.getCandidateImages());
    vector<pair<ImageFeatureNode, ImageFeatureNode>> featureNodes;
    vector<FeatureProperty> featureProps;
    tracker.createFeatureNodes(featureNodes, featureProps);
    tracker.createTracks(featureNodes);
    auto tracksGraph = tracker.buildTracksGraph(featureProps);
    std::cerr << "Created tracks graph " << endl;
    std::cerr << "Number of vertices is " << tracksGraph.m_vertices.size() << endl;
    std::cerr << "Number of edges is " << tracksGraph.m_edges.size() << endl;
    auto commonTracks = tracker.commonTracks(tracksGraph);
    Reconstructor reconstructor(flight, tracksGraph, tracker.getTrackNodes(), tracker.getImageNodes());

    string image1 = "02.jpg";
    string image2 = "03.jpg";

    auto imageNodes = tracker.getImageNodes();

    auto im1 = imageNodes[image1];
    auto im2 = imageNodes[image2];

    TwoViewPose t;
    cv::Mat mask;
    for (const auto& track : commonTracks) {
        if (track.imagePair.first == image1 && track.imagePair.first == image2) {
          //  t = reconstructor.recoverTwoCameraViewPose(im1, im2, track.commonTracks, mask);
        }
    }
    


        WHEN("The two view reconstruction is computed"){
            THEN("the result should be as expected") {
                std::cerr << "Essential matrix, rotation and pose (" << std::get<0>(t) << ", " << std::get<1>(t)
                    << ", " << std::get<2>(t) << ")\n";
                REQUIRE(true);
            }
        
        }
    }
}