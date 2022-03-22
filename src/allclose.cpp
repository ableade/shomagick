#include "allclose.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>

using std::vector;
using Eigen::Vector3d;
using cv::InputArray;
using cv::Point2d;
using cv::Mat_;

bool allClose(
    const Vector3d& lhs,
    const Vector3d& rhs,
    const double tolerance
)
{
    for (auto i = 0; i < 3; i++) {
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
    const InputArray& lhs,
    const InputArray& rhs,
    const double tolerance
) {
    const auto lhsHeader = lhs.getMat().reshape(1,1);
    const auto rhsHeader = rhs.getMat().reshape(1,1);
   
    //Assert mats  are equal size and type
    CV_Assert(
        lhsHeader.size() == rhsHeader.size() &&
        lhsHeader.type() == rhsHeader.type()
    );

    return allClose_(Mat_<double>(lhsHeader), Mat_<double> (rhsHeader), tolerance);
}

namespace detail
{
    inline bool approximatelyEqual(const double lhs, const double rhs, const double tolerance)
    {
        return fabs(lhs - rhs) <= tolerance;
    }
} //namespace detail

bool allClose(
    const Point2d& lhs,
    const Point2d& rhs,
    const double tolerance
)
{
    return  detail::approximatelyEqual(lhs.x, rhs.x, tolerance) &&
            detail::approximatelyEqual(lhs.y, rhs.y, tolerance);
}

template <typename T>
bool allClose_(
    const Mat_<T>& a,
    const Mat_<T>& b,
    const double tolerance
) {
    for (auto i = 0; i < a.cols; i++) {
        const auto lhsElem = a.template at<T>(i);
        const auto rhsElem = b.template at<T>(i);

        if (fabs(lhsElem - rhsElem) > tolerance)
        {
            return false;
        }
    }
    return true;
}

namespace
{
    template <typename T>
    void ignore(const T&) {}
} //namespace

bool allClose(
    const double lhs,
    const double rhs,
    const double tolerance
) {
    ignore(tolerance);
    return allClose(vector<double>{lhs}, vector<double>{rhs});
}
