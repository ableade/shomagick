#pragma once
#include "shotracking.h"
#include "camera.h"

inline void printGraph(TrackGraph tracksGraph) {
    std::cerr << "Number of vertices is " << tracksGraph.m_vertices.size() << endl;
    std::cerr << "Number of edges is " << tracksGraph.m_edges.size() << endl;

    TrackGraph::vertex_iterator i, end;
    std::pair<TrackGraph::vertex_iterator,
        TrackGraph::vertex_iterator> vs = boost::vertices(tracksGraph);

    std::copy(vs.first, vs.second,
        std::ostream_iterator<TrackGraph::vertex_descriptor>{
        std::cout, "\n"});

    auto es = boost::edges(tracksGraph);


    std::copy(es.first, es.second,
        std::ostream_iterator<TrackGraph::edge_descriptor>{
        std::cout, "\n"});
}

inline Camera getPerspectiveCamera(float physicalLens, int height, int width, float k1, float k2) {
    const auto pixelFocal = physicalLens * width;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        pixelFocal,
        0.,
        width / 2,
        0.,
        pixelFocal,
        height / 2,
        0.,
        0.,
        1
        );

    cv::Mat dist = (cv::Mat_<double>(4, 1) <<
        k1,
        k2,
        0.,
        0.
        );

    return { cameraMatrix, dist, height, width };
}