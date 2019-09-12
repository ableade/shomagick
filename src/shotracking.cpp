#include "shotracking.h"
#include "string"
#include <vector>
#include <iostream>
#include "utilities.h"
#include <set>

using cv::DMatch;
using cv::Point2d;
using cv::Scalar;
using std::cout;
using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::to_string;
using std::vector;
using std::string;
using std::endl;
using std::cerr;

ShoTracker::ShoTracker(
    FlightSession flight,
    std::map<string,
    std::vector<string>> candidateImages
)
    : flight(flight)
    , mapOfImageNamesToCandidateImages(candidateImages)
    , imageFeatureNodes_()
    , tracks_()
    , uf()
    , imageNodes_()
    , trackNodes_()
    , imageFeatures()
{}

void ShoTracker::createFeatureNodes(vector<pair<ImageFeatureNode, ImageFeatureNode>> &allFeatures,
    vector<FeatureProperty> &props)
{
    size_t featureIndex = 0;
    cout << "Creating feature nodes" << endl;
    //Image name and corresponding keypoint index form a single node
    for (const auto&[imageName, candidateImages] : mapOfImageNamesToCandidateImages)
    {
        auto allPairMatches = this->flight.loadMatches(imageName);
        auto leftImageName = imageName;
        auto leftImageFeatures = this->_loadImageFeatures(leftImageName);
        for (const auto&[matchImageName, dMatches] : allPairMatches)
        {
            for (const auto& dMatch : dMatches)
            {
                //The left image is the query image and the right image is the train image
                auto leftFeature = make_pair(leftImageName, dMatch.queryIdx);
                auto rightFeature = make_pair(matchImageName, dMatch.trainIdx);
                auto rightImageFeature = this->_loadImageFeatures(matchImageName);
                allFeatures.push_back(make_pair(leftFeature, rightFeature));
                if (addFeatureToIndex(leftFeature, featureIndex))
                {
                    featureIndex++;
                    FeatureProperty leftProp = getFeatureProperty_(leftImageFeatures, leftFeature);
                    props.push_back(leftProp);
                }
                if (addFeatureToIndex(rightFeature, featureIndex))
                {
                    featureIndex++;
                    FeatureProperty rightProp = getFeatureProperty_(rightImageFeature, rightFeature);
                    props.push_back(rightProp);
                }
            }
        }
    }
    assert(props.size() == featureIndex);
    assert(imageFeatureNodes_.size() == reverseImageFeatureNodes_.size());
    uf = UnionFind(imageFeatureNodes_.size());
    cout << "Created a total of " << imageFeatureNodes_.size() << " feature nodes " << endl;
}

ImageFeatures ShoTracker::_loadImageFeatures(const string fileName) {
    if (imageFeatures.find(fileName) == this->imageFeatures.end()) {
        imageFeatures[fileName] = this->flight.loadFeatures(fileName);
    }
    return imageFeatures[fileName];
}

void ShoTracker::createTracks(const vector<pair<ImageFeatureNode, ImageFeatureNode>> &features)
{
    cerr << "Creating tracks" << endl;
    for (const auto&[leftFeature, rightFeature] : features)
    {
        const auto leftFeatureGlobalID = imageFeatureNodes_.at(leftFeature);
        const auto rightFeatureGlobalID = imageFeatureNodes_.at(rightFeature);
        mergeFeatureTracks(leftFeature, rightFeature);
    }
    cerr << "Created a total of " << uf.numDisjointSets() << " tracks" << endl;

    //Filter out bad tracks
    for (const auto&[imageFeatureNode, index] : imageFeatureNodes_)
    {
        int dSet = uf.findSet(index);
        if (uf.sizeOfSet(dSet) >= this->minTrackLength)
        {
            tracks_[dSet].push_back(index);
        }
    }

    set<int> badTracks;
    for (const auto &[trackId, trackSet] : tracks_) {
        std::set<string> images;
        for (auto imFeature : trackSet) {
            const auto featureNode = retrieveFeatureByIndexValue(imFeature);
            auto[_, inserted] = images.insert(featureNode.first);
            if (inserted == false) {
                badTracks.insert(trackId);
                break;
            }
        }
    }
    for (auto badTrack : badTracks) {
        tracks_.erase(badTrack);
    }

    cerr << "Found a total of " << tracks_.size() << " good tracks " << endl;
}

TrackGraph ShoTracker::buildTracksGraph(const std::vector<FeatureProperty> &props)
{
    TrackGraph tg;
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it)
    {
        TrackGraph::vertex_descriptor track;
        auto trackSet = it->second;
        auto trackId = to_string(it->first); //Track id is the parent set of this feature set
        track = boost::add_vertex(tg);
        tg[track].name = trackId;
        tg[track].is_image = false;
        trackNodes_[trackId] = track;
        for (size_t i = 0; i < trackSet.size(); ++i)
        {
            TrackGraph::vertex_descriptor imageNode;
            auto feature = retrieveFeatureByIndexValue(trackSet[i]);
            auto prop = props[trackSet[i]];
            auto imageName = feature.first;
            if (imageNodes_.find(imageName) == imageNodes_.end())
            {
                imageNode = boost::add_vertex(tg);
                imageNodes_[imageName] = imageNode;
                tg[imageNode].is_image = true;
                tg[imageNode].name = imageName;
            }
            else
            {
                imageNode = imageNodes_[imageName];
            }
            auto e = EdgeProperty(prop, trackId, feature.first);
            boost::add_edge(imageNode, track, e, tg);
        }
    }
    return tg;
}

vector<CommonTrack> ShoTracker::commonTracks(const TrackGraph &tg) const
{
    vector<CommonTrack> commonTracks;
    map<pair<string, string>, std::set<string>> _commonTracks;
    for (auto& trackNode : trackNodes_)
    {
        std::string vertexName;
        TrackGraph::vertex_descriptor trackDescriptor;
        std::tie(vertexName, trackDescriptor) = trackNode;
        vector<string> imageNeighbours;

        auto neighbours = boost::adjacent_vertices(trackDescriptor, tg);
        for (auto vd : make_iterator_range(neighbours))
        {
            imageNeighbours.push_back(tg[vd].name);
        }
        auto combinations = _getCombinations(imageNeighbours);

        for (const auto &combination : combinations)
        {
            _commonTracks[combination].insert(vertexName);
        }
    }
    for (auto pair : _commonTracks) {
        CommonTrack cTrack(pair.first, 0, pair.second);
        commonTracks.push_back(cTrack);
    }
    return commonTracks;
}

FeatureProperty ShoTracker::getFeatureProperty_(const ImageFeatures &imageFeatures, ImageFeatureNode fNode) const
{
    assert(fNode.second < imageFeatures.keypoints.size());
    return {
        fNode,
        imageFeatures.keypoints[fNode.second].pt,
        imageFeatures.colors[fNode.second],
        imageFeatures.keypoints[fNode.second].size
    };
}

set<pair<string, string>> ShoTracker::_getCombinations(const vector<string> &images) const
{
    set<pair<string, string>> combinations;
    for (size_t i = 0; i < images.size() - 1; ++i) {
        for (size_t j = i + 1; j < images.size(); ++j) {
            auto pair = make_pair(images[i], images[j]);
            combinations.insert(pair);
        }
    }
    return combinations;
}

void ShoTracker::mergeFeatureTracks(pair<string, int> feature1, pair<string, int> feature2)
{
    uf.unionSet(imageFeatureNodes_.at(feature1), imageFeatureNodes_.at(feature2));
}

bool ShoTracker::addFeatureToIndex(pair<string, int> feature, int featureIndex)
{
    auto insert = imageFeatureNodes_.insert(make_pair(feature, featureIndex));
    const auto wasAdded = insert.second;
    if (wasAdded) {
        reverseImageFeatureNodes_.insert(make_pair(featureIndex, feature));
    }
    return wasAdded;
}

map <int, vector <int>> ShoTracker::getTracks()
{
    return tracks_;
}

std::pair<string, int> ShoTracker::retrieveFeatureByIndexValue(int index)
{
    CV_Assert(index < imageFeatureNodes_.size());
    return reverseImageFeatureNodes_[index];
}

const std::map<string, TrackGraph::vertex_descriptor> ShoTracker::getImageNodes() const {
    return imageNodes_;
}

const std::map<string, TrackGraph::vertex_descriptor> ShoTracker::getTrackNodes() const {
    return trackNodes_;
}