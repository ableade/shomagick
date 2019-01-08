#include "stdafx.h"
#include <Eigen/Core>
#include "catch.hpp"
#include <vector>
#include "../src/camera.h"
#include "../src/allclose.h"

using std::vector;
namespace
{
    template <typename T>
    struct PrintableVec
    {
        const vector<T>& vec_;
    };

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const PrintableVec<T>& printableVec)
    {
        os << "{ ";
        std::string separator;
        for (const auto& e : printableVec.vec_)
        {
            os << separator << e;
            separator = ", ";
        }
        os << " }";
        return os;
    }

    template <typename T>
    PrintableVec<T> makePrintable(const vector<T>& vec)
    {
        return PrintableVec<T>{ vec };
    }
} //namespace


Camera getPerspectiveCamera() {
    const auto physicalLens = 0.6;
    const auto height = 600;
    const auto width = 800;
    const auto dist1 = -0.1;
    const auto dist2 = 0.01;

    const auto pixelFocal = physicalLens * width;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        pixelFocal,
        0.,
        width/2,
        0.,
        pixelFocal,
        height/2,
        0.,
        0.,
        1
    );

    cv::Mat dist = (cv::Mat_<double>(4, 1) <<
        dist1,
        dist2,
        0.,
        0.
    );

    return { cameraMatrix, dist, height, width };
}

SCENARIO("Testing the projection for a perspective camera")
{
    GIVEN("a perspective camera and pixel [0.1,0.2]  ")
    {
        auto c = getPerspectiveCamera();
        WHEN("the camera projects bearing vector for this pixel")
        {
            const auto testPoint = cv::Point2f{ 0.1,0.2 };
            auto bearing = c.normalizedPointToBearingVec(
                testPoint
            );
            std::cout << "Bearing is " << bearing << std::endl;
            auto projected = c.projectBearing(bearing);
            THEN("the recovered 2d point should be close to the original")
            {
                INFO("testPoint: " << testPoint);
                INFO("projected: " << projected);
                REQUIRE(allClose(testPoint, projected));
    
            }
        }
    }
}
