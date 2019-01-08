#pragma once

template <typename Container>
bool allClose(
    const Container& lhs,
    const Container& rhs,
    const double tolerance = 1e-08
)
{
    if (lhs.size() != rhs.size())
    {
        throw std::invalid_argument{ "lhs and rhs must be the same size!" };
    }

    return std::equal(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        [tolerance](const auto lhsElem, const auto rhsElem) {
            return fabs(lhsElem - rhsElem) <= tolerance;
        }
    );
}

#include <Eigen/Core>

template <>
inline bool allClose<Eigen::Vector3d>(
    const Eigen::Vector3d& lhs,
    const Eigen::Vector3d& rhs,
    const double tolerance
)
{
    for (int i = 0; i < 3; i++) {
        const auto lhsElem = lhs(i);
        const auto rhsElem = rhs(i);
        if (fabs(lhsElem - rhsElem) > tolerance)
        {
            return false;
        }
    }
    return true;
}

#include <opencv2/core.hpp>

namespace detail
{
    inline bool approximatelyEqual( const double lhs, const double rhs, const double tolerance )
    {
        return fabs(lhs - rhs) <= tolerance;
    }
} //namespace detail

template <>
inline bool allClose<cv::Point2f>(
    const cv::Point2f& lhs,
    const cv::Point2f& rhs,
    const double tolerance
)
{
    return  detail::approximatelyEqual( lhs.x, rhs.x, tolerance ) &&
            detail::approximatelyEqual( lhs.y, rhs.y, tolerance );
}