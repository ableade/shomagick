//#include "stdafx.h"
#include <Eigen/Core>
#include <catch.hpp>
#include <vector>
#include "camera.h"
#include "shot.h"
#include "allclose.h"
#include "utilities.h"
#include <opencv2/core/eigen.hpp>

using std::vector;
using cv::Mat;
using cv::Vec3d;
using Eigen::Matrix3d;
using Eigen::Vector3d;

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


SCENARIO("Testing the projection for a perspective camera")
{
    GIVEN("a perspective camera and pixel [0.1,0.2]  ")
    {
        const auto physicalLens = 0.63;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;

        auto c = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);
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

SCENARIO("Testing the bearing direction of the ray that hits the principal point")
{
    GIVEN("a perspective camera and pixel [0.0,0.0]  ")
    {
        const auto testPoint = cv::Point2d{ 0.0,0.0 };
        const auto physicalLens = 0.6;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;

        auto c = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);
        WHEN("the bearing direction of the center pixel is calculated")
        {
            const auto expectedBearing = Eigen::Vector3d{ 0,0,1 };
            const auto actualBearing = c.normalizedPointToBearingVec(
                testPoint
            );

            THEN("the bearing vector should be [0,0,1]")
            {
                INFO("expected: " << expectedBearing);
                INFO("actual: " << actualBearing);
                REQUIRE(allClose(expectedBearing, actualBearing));

            }
        }
    }
}

SCENARIO("Testing the bearing direction of a camera at [0.03, 0.04]")
{
    GIVEN("a perspective camera and pixel [0.0,0.0]  ")
    {
        const auto testPoint = cv::Point2d{ 0.03,0.04};
        const auto physicalLens = 0.6;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;

        auto c = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);
        WHEN("the bearing direction of the center pixel is calculated")
        {
            const auto expectedBearing = Eigen::Vector3d{0.0498616991051048,  0.066482265473473,   0.9965409772507433 };
            const auto actualBearing = c.normalizedPointToBearingVec(
                testPoint
            );

            THEN("the bearing vector should be [0,0,1]")
            {
                INFO("expected: " << expectedBearing);
                INFO("actual: " << actualBearing);
                REQUIRE(allClose(expectedBearing, actualBearing));

            }
        }
    }
}

    SCENARIO("Testing the inverse of a pose")
{
    GIVEN("a pose with rotation vector [1,2,3] and translation vector [4,5,6]")
    {
        cv::Mat rotation = (cv::Mat_<double>(3, 1) << 1,2,3);
        Vec3d translation { 4, 5, 6 };
        Pose p{rotation, translation };
        const Vec3d expected { 0, 0, 0 };
        WHEN("the inverse of this pose is calculated")
        {
            const auto inv = p.poseInverse();
            std::cout << "Inverse "<< inv << std::endl;
            const auto identity = p.compose(inv);
            std::cout << "Identity " << identity << std::endl;
            THEN("the inverted pose should be as expected")
            {
                REQUIRE(allClose(identity.getRotationVector(), expected));

            }
        }
    }
}

SCENARIO("Testing the origin of a pose")
{
    GIVEN("a pose with rotation vector [1,2,3] and translation vector [4,5,6]")
    {
        Mat rotation = (cv::Mat_<double>(3, 1) << 1, 2, 3);
        Vec3d translation{ 4, 5, 6 };
        Pose p{rotation, translation};
        WHEN("the origin of this pose is calculated")
        {
            const auto actual = p.getOrigin();
            THEN("the inverted pose should be as expected")
            {
                Mat expected = (cv::Mat_<double>(3, 1) << -0.41815191, -5.12325694, -7.11177807);
                std::cout << "Origin of this pose is " << p.getOrigin()<< std::endl;
                WARN("expected: " << expected);
                WARN("actual: " << actual);
                REQUIRE(allClose(expected, actual));
            }
        }
    }
}

SCENARIO("Testing the rotation inverse of a shot")
{
    GIVEN("a shot and pose with rotation vector [1,2,3] and translation vector [4,5,6] and \
        a test camera")
    {
        const auto testPoint = cv::Point2d{ 0.23,0.678 };
        const auto physicalLens = 0.6;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;

        Mat rotation = (cv::Mat_<double>(3, 1) << 1, 2, 3);
        Mat translation = (cv::Mat_<double>(3, 1) << 4, 5, 6);
        Pose p{ rotation, translation };
        const std::string image = "im1";
        const auto camera = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);
        Shot shot{ image, camera, p };
        WHEN("the inverse of this pose is calculated")
        {
            Matrix3d eigenRotationInverse;
            const auto bearing = shot.getCamera().normalizedPointToBearingVec(testPoint);
            const auto rotationInverse = shot.getPose().getRotationMatrixInverse();
            cv2eigen(rotationInverse, eigenRotationInverse);
            Mat expectedRotationInverse = (cv::Mat_<double>(3,3) <<
                -0.6949205576413117, -0.1920069727919994, 0.6929781677417702,
                 0.7135209905277877, -0.3037850443394704, 0.6313496993837179,
                 0.0892928588619122,  0.933192353823647,  0.3481074778302649
            );
            THEN("the inverted pose should be as expected")
            {
                CAPTURE(expectedRotationInverse);
                CAPTURE(rotationInverse);
                REQUIRE(allClose(expectedRotationInverse, rotationInverse));
            }
        }
    }
}

SCENARIO("Testing the rotation inverse times bearing of a shot")
{
    GIVEN("a shot and pose with rotation vector [1,2,3] and translation vector [4,5,6] and a test camera")
    {
        const auto testPoint = cv::Point2d{ 0.23,0.678 };
        const auto physicalLens = 0.6;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;

        Mat rotation = (cv::Mat_<double>(3, 1) << 1, 2, 3);
        Vec3d translation{ 4, 5, 6 };
        Pose p{ rotation, translation };
        const std::string image = "im1";
        const auto camera = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);
        Shot shot{ image, camera, p };
        WHEN("the inverse of this pose is calculated")
        {
            Matrix3d eigenRotationInverse;
            const auto bearing = shot.getCamera().normalizedPointToBearingVec(testPoint);
            const auto rotationInverse = shot.getPose().getRotationMatrixInverse();
            cv2eigen(rotationInverse, eigenRotationInverse);
            THEN("the inverted pose should be as expected")
            {
                const auto actual = eigenRotationInverse * bearing;
                Vector3d expected (0.06710956564396,  0.3152355286551768,  0.9466376644062762);
                CAPTURE(expected);
                CAPTURE(actual);
                REQUIRE(allClose(expected, actual, 1e-7));
            }
        }
    }
}
SCENARIO("Testing normalized point conversion")
{
    GIVEN("a point, test camera with width 800 and height 600")
    {
        const vector<cv::Point2d> inputPoints {
            {   0, 0 },
            { 319, 240 },
            { 800, 600 }
        };


        const vector<cv::Point2d> expectedPoints{
            { -0.4993750000000, -0.3743750000000 },
            { -0.1006250000000, -0.0743750000000 },
            {  0.5006250000000,	 0.3756250000000 }
        };

        using InputPoint = cv::Point2d;
        using ExpectedPoint = cv::Point2d;
        using InputWithExpectedPoint = std::pair<InputPoint, ExpectedPoint>;
        const auto inputsWithExpected = std::vector<InputWithExpectedPoint>{
            { {   0,   0 }, { -0.4993750000000, -0.3743750000000 } },
            { { 319, 240 }, { -0.1006250000000, -0.0743750000000 } },
            { { 800, 600 }, {  0.5006250000000,	 0.3756250000000 } },
        };

        const auto physicalLens = 0.6;
        const auto height = 600;
        const auto width = 800;
        const auto dist1 = -0.1;
        const auto dist2 = 0.01;
        const auto camera = getPerspectiveCamera(physicalLens, height, width, dist1, dist2);

        WHEN("the normalized point coordinate is calculated") {
            const auto normalizedPoints = camera.normalizeImageCoordinates(inputPoints);
            THEN("The result should be as expected") {
                CAPTURE(expectedPoints);
                CAPTURE(normalizedPoints);
                REQUIRE(allClose(expectedPoints, normalizedPoints));
            }
           
        }
    }
    
}