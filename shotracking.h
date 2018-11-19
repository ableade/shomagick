#ifndef SHO_TRACKING_HPP__
#define SHO_TRACKING_HPP__

#include "unionfind.h"
#include "flightsession.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>

struct VertexProperty
{
	string name;
	bool is_image;
};

typedef boost::adjacency_list<boost::listS, boost::setS, boost::undirectedS, VertexProperty, boost::no_property> TrackGraph; 


class ShoTracker
{
private:
	FlightSession flight;
	std::map<string, std::vector<string>> candidateImages;
	std::map<string, std::vector<cv::DMatch>> candidateMatches;
	std::map <std::pair <string, int>, int> features;
	std::map <int, std::vector <int>> tracks;
	UnionFind uf;
	int minTrackLength =2;
	bool addFeatureToIndex(std::pair <string, int> feature, int featureIndex);
	std::map <string, TrackGraph::vertex_descriptor> imageNodes;
	std::map <string, TrackGraph::vertex_descriptor> trackNodes;
	std::set<std::pair < string , string> > getCombinations_ (std::vector< string> images);
	
public:
	ShoTracker(FlightSession flight, std::map<string, std::vector<string>> candidateImages);
	void createTracks();
	TrackGraph buildTracksGraph();
	void mergeFeatureTracks(std::pair <string, int> feature1, std::pair <string, int> feature2);
	void createFeatureNodes();
	std::map<std::pair < string, string>, std::vector<string> > commonTracks(const TrackGraph& tg);
	std::map <int, std::vector <int>> getTracks();
	std::pair<string, int> retrieveFeatureByIndexValue(int index);
	std::map <string, TrackGraph::vertex_descriptor> getTrackNodes();
	std::map <string, TrackGraph::vertex_descriptor> getImageNodes();
};
#endif