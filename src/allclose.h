#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>

static constexpr auto defaultTolerance = 1e-08;

bool allClose(
    const Eigen::Vector3d& lhs,
    const Eigen::Vector3d& rhs,
    const double tolerance = defaultTolerance
);

bool allClose(
    const cv::InputArray& lhs,
    const cv::InputArray& rhs,
    const double tolerance = defaultTolerance
);

bool allClose(
    const cv::Point2d& lhs,
    const cv::Point2d& rhs,
    const double tolerance = defaultTolerance
);

bool allClose(
    const double lhs,
    const double rhs,
    const double tolerance = defaultTolerance
);

template <typename T>
bool allClose_(
    const cv::Mat_<T>& a, 
    const cv::Mat_<T>& b, 
    const double tolerance = defaultTolerance
);


template<typename DerivedA, typename DerivedB>
bool allCloseEigen(
    const Eigen::DenseBase<DerivedA>& a,
    const Eigen::DenseBase<DerivedB>& b,
    const typename DerivedA::RealScalar& rtol
        = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar& atol
        = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()
)
{
    return ((a.derived() - b.derived()).array().abs()
        <= (atol + rtol * b.derived().array().abs())).all();
}
