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

	VertexProperty() : name() , is_image() {};
	VertexProperty(string name, bool is_image) : name(name), is_image(is_image) {};

};

struct FeatureProperty
{
	cv::Point2f coordinates;
	cv::Scalar color;

	FeatureProperty() : coordinates(), color() {};
	FeatureProperty(cv::Point2f coordinates, cv::Scalar color) : coordinates(coordinates) , color(color) {}
};

struct EdgeProperty {
	FeatureProperty fProp;
	string trackName;
	string imageName;

	EdgeProperty() : fProp(), trackName(), imageName() {}
	EdgeProperty(FeatureProperty fProp, string trackName, string imageName) : fProp(fProp), trackName(trackName), imageName(imageName) {} 
};

typedef boost::adjacency_list<boost::listS, boost::setS, boost::undirectedS, VertexProperty, EdgeProperty> TrackGraph;
typedef boost::graph_traits <TrackGraph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits <TrackGraph>::adjacency_iterator adjacency_iterator;
typedef std::pair<string, int> FeatureNode;

class CommonTrack {
	public:
		std::pair<string, string> imagePair;
		float rScore;
		std::set <string> commonTracks;

		CommonTrack() : imagePair(), rScore(), commonTracks() {};
		CommonTrack(std::pair<string, string> imagePair, float rScore, std::set<string> commonTracks) : imagePair
		(imagePair),  rScore(rScore), commonTracks(commonTracks) {};
};

class ShoTracker
{
  private:
	FlightSession flight;
	std::map<string, std::vector<string>> candidateImages;
	std::map<FeatureNode, int> features;
	std::map<int, std::vector<int>> tracks;
	UnionFind uf;
	int minTrackLength = 2;
	bool addFeatureToIndex(std::pair<string, int> feature, int featureIndex);
	std::map<string, TrackGraph::vertex_descriptor> imageNodes;
	std::map<string, TrackGraph::vertex_descriptor> trackNodes;
	std::set<std::pair<string, string>> _getCombinations(const std::vector<string>& images) const;
	FeatureProperty _getFeatureProperty (const ImageFeatures& imageFeatures, int featureIndex);

  public:
	ShoTracker(FlightSession flight, std::map<string, std::vector<string>> candidateImages);
	void createTracks(const std::vector<std::pair<FeatureNode, FeatureNode>>& features);
	TrackGraph buildTracksGraph(const std::vector<FeatureProperty>& props);
	void mergeFeatureTracks(FeatureNode feature1, FeatureNode feature2);
	void createFeatureNodes(std::vector<std::pair<FeatureNode, FeatureNode>>& allFeatures, 
	std::vector<FeatureProperty> & props);
	std::vector<CommonTrack> commonTracks(const TrackGraph &tg) const;
	std::map <int, std::vector <int>> getTracks();
	FeatureNode retrieveFeatureByIndexValue(int index);
	const std::map<string, TrackGraph::vertex_descriptor> getTrackNodes() const;
	const std::map<string, TrackGraph::vertex_descriptor> getImageNodes() const;
};
#endif