#include "shotracking.h"
#include "string"
#include <vector>
#include <iostream>
#include <set>

using cv::DMatch;
using cv::Point2f;
using cv::Scalar;
using std::cout;
using std::make_pair;
using std::map;
using std::pair;
using std::set;
using std::to_string;
using std::vector;

ShoTracker::ShoTracker(
    FlightSession flight,
    std::map<string,
    std::vector<string>> candidateImages
)
    : flight(flight)
    , candidateImages(candidateImages)
    , features()
    , tracks()
    , uf()
    , imageNodes()
    , trackNodes()
{}

void ShoTracker::createFeatureNodes(vector<pair<FeatureNode, FeatureNode>> &allFeatures,
                                                                      vector<FeatureProperty> &props)
{
    size_t featureIndex = 0;
    cout << "Creating feature nodes" << endl;
    //Image name and corresponding keypoint index form a single node
    for (auto it = this->candidateImages.begin(); it != this->candidateImages.end(); ++it)
    {
        auto allPairMatches = this->flight.loadMatches(it->first);
        auto leftImageName = it->first;
        auto leftImageFeatures = this->flight.loadFeatures(leftImageName);
        for (auto matchIt = allPairMatches.begin(); matchIt != allPairMatches.end(); ++matchIt)
        {
            auto matchImageName = matchIt->first;
            for (size_t k = 0; k < matchIt->second.size(); ++k)
            {
                auto leftFeature = make_pair(leftImageName, matchIt->second[k].trainIdx);
                auto rightFeature = make_pair(matchImageName, matchIt->second[k].queryIdx);
                auto rightImageFeature = this->flight.loadFeatures(matchImageName);
                if (this->addFeatureToIndex(leftFeature, featureIndex))
                {
                    featureIndex++;
                    FeatureProperty leftProp = this->_getFeatureProperty(leftImageFeatures, matchIt->second[k].trainIdx);
                    props.push_back(leftProp);
                    if (this->addFeatureToIndex(rightFeature, featureIndex))
                    {
                        featureIndex++;
                        allFeatures.push_back(make_pair(leftFeature, rightFeature));
                        FeatureProperty rightProp = this->_getFeatureProperty(rightImageFeature, matchIt->second[k].queryIdx);
                        props.push_back(rightProp);
                    }
                }
            }
        }
        assert(props.size() == featureIndex);
    }
    this->uf = UnionFind(this->features.size());
    cout << "Created a total of " << this->features.size() << " feature nodes " << endl;
}

void ShoTracker::createTracks(const vector<pair<FeatureNode, FeatureNode>> &features)
{
    cout << "Creating tracks" << endl;
    for (size_t i = 0; i < features.size(); ++i)
    {
        this->mergeFeatureTracks(features[i].first, features[i].second);
    }
    cout << "Created a total of " << this->uf.numDisjointSets() << " tracks " << endl;

    //Filter out bad tracks
    for (size_t i = 0; i < this->features.size(); ++i)
    {
        int dSet = this->uf.findSet(i);
        if (this->uf.sizeOfSet(dSet) >= this->minTrackLength)
        {
            this->tracks[dSet].push_back(i);
        }
    }

    cout << "Found a total of " << this->tracks.size() << " good tracks " << endl;
}

TrackGraph ShoTracker::buildTracksGraph(const std::vector<FeatureProperty> &props)
{
    TrackGraph tg;
    for (auto it = this->tracks.begin(); it != this->tracks.end(); ++it)
    {
        TrackGraph::vertex_descriptor track;
        auto trackSet = it->second;
        auto trackId = to_string(it->first); //Track id is the parent set of this feature set
        track = boost::add_vertex(tg);
        tg[track].name = trackId;
        tg[track].is_image = false;
        this->trackNodes[trackId] = track;
        for (size_t i = 0; i < trackSet.size(); ++i)
        {
            TrackGraph::vertex_descriptor imageNode;
            auto feature = this->retrieveFeatureByIndexValue(trackSet[i]);
            auto prop = props[trackSet[i]];
            auto imageName = feature.first;
            if (this->imageNodes.find(imageName) == this->imageNodes.end())
            {
                imageNode = boost::add_vertex(tg);
                this->imageNodes[imageName] = imageNode;
                tg[imageNode].is_image = true;
                tg[imageNode].name = imageName;
            }
            else
            {
                imageNode = this->imageNodes[imageName];
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
    map<pair<string, string>, std::vector<string>> _commonTracks;
    for (auto it = this->trackNodes.begin(); it != this->trackNodes.end(); ++it)
    {
        vector<string> imageNeighbours;
        auto neighbours = boost::adjacent_vertices(it->second, tg);
        for (auto vd : make_iterator_range(neighbours))
        {
            imageNeighbours.push_back(tg[vd].name);
        }
        auto combinations = this->_getCombinations(imageNeighbours);
        
        for (auto combination : combinations)
        {
            _commonTracks[combination].push_back(it->first);
        }
    }
    commonTracks.reserve(_commonTracks.size());
    for(auto pair : _commonTracks) {
        set <string> trackset(pair.second.begin(), pair.second.end());
        CommonTrack cTrack(pair.first, 0, trackset);
        commonTracks.push_back(cTrack);
    }
    return commonTracks;
}

FeatureProperty ShoTracker::_getFeatureProperty(const ImageFeatures &imageFeatures, int featureIndex)
{
    return {imageFeatures.keypoints[featureIndex].pt, imageFeatures.colors[featureIndex]};
}

set<pair<string, string>> ShoTracker::_getCombinations(const vector<string> &images) const
{
    auto r = 2;
    std::vector<bool> v(images.size());
    set<pair<string, string>> combinations;
    std::fill(v.begin(), v.begin() + r, true);
    do
    {
        std::vector<string> aPair;
        for (size_t i = 0; i < images.size(); ++i)
        {
            if (v[i])
            {
                aPair.push_back(images[i]);
            }
        }
        combinations.insert(make_pair(aPair[0], aPair[1]));
    } while (std::prev_permutation(v.begin(), v.end()));
    return combinations;
}

void ShoTracker::mergeFeatureTracks(pair<string, int> feature1, pair<string, int> feature2)
{
    this->uf.unionSet(this->features[feature1], this->features[feature2]);
}

bool ShoTracker::addFeatureToIndex(pair<string, int> feature, int featureIndex)
{
    auto insert = this->features.insert(make_pair(feature, featureIndex));
    return insert.second;
}

map <int, vector <int>> ShoTracker::getTracks()
{
    return this->tracks;
}

std::pair<string, int> ShoTracker::retrieveFeatureByIndexValue(int index)
{
    for (auto it = this->features.begin(); it != this->features.end(); ++it)
    {
        if (it->second == index)
        {
            return it->first;
        }
    }
    return pair<string, int>();
}

const std::map<string, TrackGraph::vertex_descriptor> ShoTracker::getImageNodes() const {
    return this->imageNodes;
}

const std::map<string, TrackGraph::vertex_descriptor> ShoTracker:: getTrackNodes() const {
    return this->trackNodes;
}