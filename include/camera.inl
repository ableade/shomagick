#include <algorithm>

template <typename T>
inline std::vector<cv::Point_<T>> Camera::normalizeImageCoordinates(const std::vector<cv::Point_<T>>& points) const
{
    std::vector<cv::Point_<T>> results;

    for (const auto& point : points) {
        results.push_back(normalizeImageCoordinate(point));
    }
    
    return results;
}

template <typename T>
inline cv::Point_<T> Camera::normalizeImageCoordinate(const cv::Point_<T>& pixelCoords) const
{
    auto h = (scaledHeight_) ? scaledHeight_ : height_;
    auto w = (scaledWidth_) ? scaledWidth_ : width_;

    const auto size = std::max(w, h);

    float step = 0.5;
    const auto pixelX = pixelCoords.x + step;
    const auto pixelY = pixelCoords.y + step;
    const auto normX = ((1.0f * w) / size) * (pixelX - w / 2.0f) / w;
    const auto normY = ((1.0f * h) / size) * (pixelY - h / 2.0f) / h;

    return {
        normX,
        normY,
    };
}

template <typename T>
inline opengv::bearingVectors_t Camera::normalizedPointsToBearingVec(const std::vector<cv::Point_<T>>& points) const
{
    opengv::bearingVectors_t bearings;
    for (const auto point : points) {
        auto bearing = normalizedPointToBearingVec(point);
        bearings.push_back(bearing);
    }
    return bearings;
}

template <typename T>
inline opengv::bearingVector_t Camera::normalizedPointToBearingVec(const cv::Point_<T> &point) const
{
    std::vector<cv::Point2d> points{ point };
    std::vector<cv::Point3d> hPoints;
    cv::undistortPoints(points, points, getNormalizedKMatrix(), getDistortionMatrix());
    cv::convertPointsHomogeneous(points, hPoints);
    opengv::bearingVector_t bearing;
    auto convPoint = hPoints[0];
    auto hPoint = cv::Vec3d(convPoint);
    const double l = std::sqrt(hPoint[0] * hPoint[0] + hPoint[1] * hPoint[1] + hPoint[2] * hPoint[2]);
    for (int j = 0; j < 3; ++j)
        bearing[j] = hPoint[j] / l;

    return bearing;
}

template<typename T>
inline cv::Point_<T> Camera::denormalizeImageCoordinates(const cv::Point_<T>& normalizedCoords) const
{
    auto h = (scaledHeight_) ? scaledHeight_ : height_;
    auto w = (scaledWidth_) ? scaledWidth_ : width_;

    const auto size = std::max(w, h);
    auto normX = normalizedCoords.x;
    auto normY = normalizedCoords.y;

    float pixelX = ((normX * width_  * size / (1.0f * w)) + w / 2.0f) - 0.5;
    float pixelY = ((normY * height_ * size / (1.0f * h)) + h / 2.0f) - 0.5;

    return { pixelX, pixelY };
}

template<typename T>
inline std::vector<cv::Point_<T>> Camera::denormalizeImageCoordinates(const std::vector<cv::Point_<T>>& points) const
{
    std::vector<cv::Point2d> results;
    for (const auto point : points) {
        results.push_back(denormalizeImageCoordinates(point));
    }
    return results;
}
