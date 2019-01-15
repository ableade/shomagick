#include "allclose.h"
#include <vector>
#include <iostream>
#include <algorithm> //std::equal
#include <Eigen/Core>

bool allClose(
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


bool allClose(
    const cv::InputArray& lhs,
    const cv::InputArray& rhs,
    const double tolerance
) {
    const auto lhsHeader = lhs.getMat().reshape(1,1);
    const auto rhsHeader = rhs.getMat().reshape(1,1);
   
    //Assert mats  are equal size and type
    CV_Assert(
        lhsHeader.size() == rhsHeader.size() &&
        lhsHeader.type() == rhsHeader.type()
    );

    return allClose_(cv::Mat_<double>(lhsHeader), cv::Mat_<double> (rhsHeader), tolerance);
}

namespace detail
{
    inline bool approximatelyEqual(const double lhs, const double rhs, const double tolerance)
    {
        return fabs(lhs - rhs) <= tolerance;
    }
} //namespace detail

bool allClose(
    const cv::Point2f& lhs,
    const cv::Point2f& rhs,
    const double tolerance
)
{
    return  detail::approximatelyEqual(lhs.x, rhs.x, tolerance) &&
            detail::approximatelyEqual(lhs.y, rhs.y, tolerance);
}

template <typename T>
bool allClose_(
    const cv::Mat_<T>& a,
    const cv::Mat_<T>& b,
    const double tolerance
) {
    for (int i = 0; i < a.cols; i++) {
        const auto lhsElem = a.at<T>(i);
        const auto rhsElem = b.at<T>(i);

        if (fabs(lhsElem - rhsElem) > tolerance)
        {
            return false;
        }
    }
    return true;
}
