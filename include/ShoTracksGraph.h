#pragma once
#include <string>
#include <map>
#include "shotracking.h"

class ShoTracksGraph {
public:
    using TrackName = std::string;
    using ImageName = std::string;
    using TrackNodes = std::map<TrackName, TrackGraph::vertex_descriptor>;
    using ImageNodes = std::map<ImageName, TrackGraph::vertex_descriptor>;

private:
    TrackGraph tg_;
    TrackNodes trackNodes_;
    ImageNodes imageNodes_;
    void _initNodes();

public:
    ShoTracksGraph() :tg_(), trackNodes_(), imageNodes_() {}
    ShoTracksGraph(TrackGraph tg);
    ShoTracksGraph (const ShoTracksGraph& stg);
    const TrackNodes& getTrackNodes() const;
    const ImageNodes& getImageNodes() const;
    const TrackGraph& getTrackGraph() const;
};