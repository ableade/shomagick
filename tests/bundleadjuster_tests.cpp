#include <catch.hpp>
#include "../src/bundle.h"


SCENARIO("Single camera for bundle adjustment")
{
    GIVEN("a perspective camera and pixel [0.1,0.2]  ")
    {
        BundleAdjuster sa;
        sa.AddShot("1", "cam1", 0.0, 0, 0, 0, 0, 0, false);
        //sa.add
        WHEN("the camera projects bearing vector for this pixel")
        {
            const auto testPoint = cv::Point2d{ 0.1,0.2 };
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