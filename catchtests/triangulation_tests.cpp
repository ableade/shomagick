#include "stdafx.h"
#include "catch.hpp"
#include <Eigen/Core>
#include <vector>
#include "../src/allclose.h"
#include "../src/shompi.h"
#include "../src/multiview.h"


namespace
{
    Eigen::Vector3d unitVector(const Eigen::Vector3d& x)
    {
        return x / x.norm();
    }
} //namespace

SCENARIO("Triangulating bearings for two generated shots")
{
    GIVEN("two shot origins, two vector bearings, max reprojection of 0.01, min ray angle of 2 degrees")
    {
        const auto o1 = Eigen::Vector3d{ 0.0, 0, 0 };
        const auto b1 = unitVector({ 0.0, 0, 1 });
        const auto o2 = Eigen::Vector3d{ 1.0, 0, 0 };
        const auto b2 = unitVector({ -1.0,0,1 });
        const auto maxReprojection = 0.01;
        const auto minRayAngle = 2.0 * M_PI / 180;

        const auto oList = std::vector<Eigen::Vector3d>{ o1,o2 };
        const auto bList = std::vector<Eigen::Vector3d>{ b1,b2 };
        WHEN("the bearings are used to triangulate a 3d point")
        {
            Eigen::Vector3d actualResult;
            auto successfullyTriangulated = csfm::TriangulateBearingsMidpoint(
                oList,
                bList,
                actualResult,
                maxReprojection,
                minRayAngle
            );

            THEN("the recovered 3d point should be close to coordinate [0,0,1.0]")
            {
                REQUIRE(successfullyTriangulated);
                const auto expectedResult = Eigen::Vector3d{ 0,0,1.0 };
                const auto expectedIsCloseToActual = allClose(expectedResult, actualResult);
                REQUIRE(expectedIsCloseToActual);
            }
        }
    }
}
