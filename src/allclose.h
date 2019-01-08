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