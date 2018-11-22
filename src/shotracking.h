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
typedef std::pair <string, int > FeatureNode;

class ShoTracker
{
private:
	FlightSession flight;
	std::map<string, std::vector<string>> candidateImages;
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
	void createTracks(std::vector < std::pair <FeatureNode, FeatureNode>> features);
	TrackGraph buildTracksGraph();
	void mergeFeatureTracks(FeatureNode feature1, FeatureNode feature2);
	std::vector< std::pair <FeatureNode, FeatureNode>> createFeatureNodes();
	std::map<std::pair < string, string>, std::vector<string> > commonTracks(const TrackGraph& tg);
	std::map <int, std::vector <int>> getTracks();
	FeatureNode retrieveFeatureByIndexValue(int index);
	std::map <string, TrackGraph::vertex_descriptor> getTrackNodes();
	std::map <string, TrackGraph::vertex_descriptor> getImageNodes();
};
#endif