#include "ShoTracksGraph.h"

void ShoTracksGraph::_initNodes()
{
    trackNodes_.clear();
    imageNodes_.clear();
    //Initialize track and image nodes using references from the track graph
    std::pair<vertex_iterator, vertex_iterator> allVertices =
        boost::vertices(tg_);
    for (; allVertices.first != allVertices.second; ++allVertices.first) {
        if (tg_[*allVertices.first].is_image) {
            imageNodes_[tg_[*allVertices.first].name] = *allVertices.first;
        }
        else {
            trackNodes_[tg_[*allVertices.first].name] = *allVertices.first;
        }
    }
}

ShoTracksGraph::ShoTracksGraph(TrackGraph tg): tg_(tg)
{
    _initNodes();
}

ShoTracksGraph::ShoTracksGraph(const ShoTracksGraph & stg)
{
    tg_ = stg.tg_;
    _initNodes();
}

const ShoTracksGraph::TrackNodes& ShoTracksGraph::getTrackNodes() const
{
    return trackNodes_;
}

const ShoTracksGraph::ImageNodes& ShoTracksGraph::getImageNodes() const
{
    return imageNodes_;
}

const TrackGraph & ShoTracksGraph::getTrackGraph() const
{
    return tg_;
}
