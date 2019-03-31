#pragma once
#include "shotracking.h"
#include "camera.h"

inline void printGraph(TrackGraph tracksGraph) {
    std::cerr << "Number of vertices is " << tracksGraph.m_vertices.size() << std::endl;
    std::cerr << "Number of edges is " << tracksGraph.m_edges.size() << std::endl;

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

inline Camera getPerspectiveCamera(float physicalLens, int height, int width, double k1, double k2) {
    const auto pixelFocal = physicalLens * std::max(height, width);

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

    return { cameraMatrix, dist , height, width };
}

template<typename AssociativeContainer, typename Predicate>
void erase_if_impl(AssociativeContainer& container, Predicate shouldRemove)
{
    for (auto it = begin(container); it != end(container); /* nothing here, the increment in dealt with inside the loop */)
    {
        if (shouldRemove(*it))
        {
            it = container.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

template<typename Key, typename Value, typename Comparator, typename Predicate>
void erase_if(std::map<Key, Value, Comparator>& container, Predicate shouldRemove)
{
    return erase_if_impl(container, shouldRemove);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <class T>
void multiplyMat_ByScalar(cv::Mat_<T> & m, T s) {
    for (auto & element : m) {
        element *= s;
    }
}

ShoRowVector3d convertPointToRowVector(cv::Point3d);
ShoRowVector3d convertVecToRowVector(cv::Vec3d);
ShoColumnVector3d convertVecToColumnVector(cv::Vec3d);
ShoRowVector4d convertPointToRowVector(cv::Vec4d);
ShoRowVector4d convertColumnVecToRowVector(ShoColumnVector3d);