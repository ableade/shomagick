#ifndef SHO_TRACKING_HPP__
#define SHO_TRACKING_HPP__

#include "unionfind.h"
#include "flightsession.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>

struct VertexProperty
{
	std::string name;
	bool is_image;

	VertexProperty() : name() , is_image() {};
	VertexProperty(std::string name, bool is_image) : name(name), is_image(is_image) {};

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
	std::string trackName;
	std::string imageName;

	EdgeProperty() : fProp(), trackName(), imageName() {}
	EdgeProperty(FeatureProperty fProp, std::string trackName, string imageName) : fProp(fProp), trackName(trackName), imageName(imageName) {} 
};

using ShoOutEdgeListS = boost::listS;
using ShoVertexListS = boost::setS;
using ShoUndirectedS = boost::undirectedS;
using ShoVertexProperty = VertexProperty;
using ShoEdgeProperty = EdgeProperty;
using ShoGraphProperty = boost::no_property;
using ShoEdgeListS = boost::listS;

using TrackGraph = boost::adjacency_list<
    ShoOutEdgeListS,
    ShoVertexListS,
    ShoUndirectedS,
    ShoVertexProperty,
    ShoEdgeProperty,
    ShoGraphProperty,
    ShoEdgeListS
>;

typedef boost::graph_traits<TrackGraph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<TrackGraph>::edge_iterator edge_iterator;
typedef boost::graph_traits<TrackGraph>::adjacency_iterator adjacency_iterator;
typedef boost::graph_traits<TrackGraph>::vertex_iterator vertex_iterator;
typedef boost::graph_traits<TrackGraph>::vertex_descriptor vertex_descriptor;
typedef std::pair<std::string, int> ImageFeatureNode;

class CommonTrack {
	public:
		std::pair<std::string, std::string> imagePair;
		float rScore;
		std::set <std::string> commonTracks;

		CommonTrack() : imagePair(), rScore(), commonTracks() {};
		CommonTrack(std::pair<string, string> imagePair, float rScore, std::set<string> commonTracks) : imagePair
		(imagePair),  rScore(rScore), commonTracks(commonTracks) {};
};

class ShoTracker
{
  private:
	FlightSession flight;
	std::map<string, std::vector<string>> candidateImages;
	std::map<ImageFeatureNode, int> features;
	std::map<int, std::vector<int>> tracks;
	UnionFind uf;
	int minTrackLength = 2;
	bool addFeatureToIndex(std::pair<string, int> feature, int featureIndex);
	std::map<std::string, TrackGraph::vertex_descriptor> imageNodes;
	std::map<std::string, TrackGraph::vertex_descriptor> trackNodes;
    std::map<std::string, ImageFeatures> imageFeatures;
	std::set<std::pair<std::string, std::string>> _getCombinations(const std::vector<std::string>& images) const;
	FeatureProperty _getFeatureProperty (const ImageFeatures& imageFeatures, int featureIndex);
    ImageFeatures _loadImageFeatures(const std::string fileName);

  public:
	ShoTracker(FlightSession flight, std::map<string, std::vector<std::string>> candidateImages);
	void createTracks(const std::vector<std::pair<ImageFeatureNode, ImageFeatureNode>>& features);
	TrackGraph buildTracksGraph(const std::vector<FeatureProperty>& props);
	void mergeFeatureTracks(ImageFeatureNode feature1, ImageFeatureNode feature2);
	void createFeatureNodes(std::vector<std::pair<ImageFeatureNode, ImageFeatureNode>>& allFeatures, 
	std::vector<FeatureProperty> & props);
	std::vector<CommonTrack> commonTracks(const TrackGraph &tg) const;
	std::map <int, std::vector <int>> getTracks();
	ImageFeatureNode retrieveFeatureByIndexValue(int index);
	const std::map<string, TrackGraph::vertex_descriptor> getTrackNodes() const;
	const std::map<string, TrackGraph::vertex_descriptor> getImageNodes() const;
};
#endif