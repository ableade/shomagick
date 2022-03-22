#pragma once

#include "unionfind.h"
#include "flightsession.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>


typedef std::string ImageName;
typedef int KeyPointIndex;
typedef std::pair<ImageName, KeyPointIndex> ImageFeatureNode;

std::set<std::pair<std::string, std::string>> _getCombinations(const std::vector<std::string>& images);

struct VertexProperty
{
    std::string name;
    bool is_image;

    VertexProperty() : name(), is_image() {};
    VertexProperty(std::string name, bool is_image) : name(name), is_image(is_image) {};

};

/*
Encapsulates all the properties of a feature in an image.
*/
struct FeatureProperty
{
    ImageFeatureNode featureNode;
    cv::Point2d coordinates;
    cv::Scalar color;
    double scale;

    FeatureProperty() : featureNode(), coordinates(), color(), scale() {};
    FeatureProperty(ImageFeatureNode fNode, cv::Point2d coordinates, cv::Scalar color, double scale) : featureNode(fNode), coordinates(coordinates), color(color), scale(scale) {}
};

/*
Encapsulates all the properties of an edge between images.
*/
struct EdgeProperty {
    FeatureProperty featureProp;
    std::string trackName;
    std::string imageName;

    EdgeProperty() : featureProp(), trackName(), imageName() {}
    EdgeProperty(FeatureProperty fProp, std::string trackName, std::string imageName) : featureProp(fProp), trackName(trackName), imageName(imageName) {}
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

class CommonTrack {
public:
    std::pair<std::string, std::string> imagePair;
    float rScore;
    std::set <std::string> commonTracks;

    CommonTrack() : imagePair(), rScore(), commonTracks() {};
    CommonTrack(std::pair<std::string, std::string> imagePair, float rScore, std::set<std::string> commonTracks) : imagePair
    (imagePair), rScore(rScore), commonTracks(commonTracks) {};

    CommonTrack(const CommonTrack& c)
    {
        imagePair = c.imagePair;
        rScore = c.rScore;
        commonTracks = c.commonTracks;
    }

    CommonTrack& operator= (const CommonTrack &c)
    {
        imagePair = c.imagePair;
        rScore = c.rScore;
        commonTracks = c.commonTracks;
        return *this;
    }
    friend std::ostream & operator << (std::ostream& out, const  CommonTrack & c) {
        out << c.imagePair.first << " - " << c.imagePair.second << "\n";
        out << "Number of common tracks is " << c.commonTracks.size() << "\n";
        out << "Score: " << c.rScore << "\n";
    }
};

class ShoTracker
{
private:
    FlightSession flight;
    using ImageName = std::string;
    using CandidateImageNames = std::vector<ImageName>;
    std::map<ImageName, CandidateImageNames> mapOfImageNamesToCandidateImages;

public:
    typedef int GloballyUniqueImageFeatureId;
private:
    std::map<ImageFeatureNode, GloballyUniqueImageFeatureId> imageFeatureNodes_;
    std::map<GloballyUniqueImageFeatureId, ImageFeatureNode> reverseImageFeatureNodes_;
    std::map<int, std::vector<int>> tracks_;
    UnionFind uf;
    int minTrackLength_ = 2;
    bool addFeatureToIndex(std::pair<std::string, int> feature, int featureIndex);
    std::map<std::string, TrackGraph::vertex_descriptor> imageNodes_;
    std::map<std::string, TrackGraph::vertex_descriptor> trackNodes_;
    std::map<std::string, ImageFeatures> imageFeatures;
    FeatureProperty getFeatureProperty_(const ImageFeatures& imageFeatures, ImageFeatureNode fNode) const;
    ImageFeatures _loadImageFeatures(const std::string fileName);

public:
    ShoTracker(FlightSession flight, std::map<std::string, std::vector<std::string>> candidateImages);
    void createTracks(const std::vector<std::pair<ImageFeatureNode, ImageFeatureNode>>& features);
    TrackGraph buildTracksGraph();
    void mergeFeatureTracks(ImageFeatureNode feature1, ImageFeatureNode feature2);
    void createFeatureNodes(std::vector<std::pair<ImageFeatureNode, ImageFeatureNode>>& allFeatures,
        std::vector<FeatureProperty> & props);
    std::map <int, std::vector <int>> getTracks();
    ImageFeatureNode retrieveFeatureByIndexValue(int index);
    const std::map<std::string, TrackGraph::vertex_descriptor> getTrackNodes() const;
    const std::map<std::string, TrackGraph::vertex_descriptor> getImageNodes() const;
};