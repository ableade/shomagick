#include "stdafx.h"
#include <opencv2/core.hpp>
#include <Eigen/core>
#include "../src/transformations.h"
#include "../src/allclose.h"
#include "catch.hpp"

using cv::Mat;
using cv::Mat_;
using cv::Point3d;
using cv::Vec3d;
using std::vector;
using cv::Vec4d;
using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Matrix;
using Eigen::Matrix4d;

namespace transformationtests
{
    SCENARIO("Test angle between two vectors")
    {
        GIVEN("Given two points [1, -2, 3], [-1, 2, -3]")
        {
            Point3d v0{ 1,-2,3 };
            Point3d v1{ -1,2,-3 };
            WHEN("We calculate the angle between these points") {
                const auto result = calculateAngleBetweenVectors(v0, v1);
                THEN("The result should be as expected") {
                    const auto expected = CV_PI;;
                    REQUIRE(allClose(result, expected));
                }
            }
        }
    }

    SCENARIO("Test angle between a vector of points and one point")
    {
        GIVEN("Given a vector of 4d points [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]] and  \
            a point [3,0.0]")
        {
            vector<Vec4d> v0{ {2, 0, 0, 2},{0, 2, 0, 2},{0, 0, 2, 2} };
            vector<double> v1{ 3,0,0 };
            WHEN("We calculate the angle between these points") {
                const Mat_<double> expected = (Mat_<double>(1, 4) << 0 ,1.57079633, 1.57079633, 0.95531662);
                const auto result = calculateAngleBetweenVectors(v0, v1);
                THEN("The result should be as expected") {
                    REQUIRE(allClose(result, expected));
                }
            }
        }
    }

    SCENARIO("Test angle between two vectors of 3d points")
    {
        GIVEN("Given a vector of 3d points [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]] and  \
           another vector of 3d points [ [0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]")
        {
            vector<Vec3d> v0{ {2, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, 0, 0} };
            vector<Vec3d> v1{ {0, 3, 0}, {0, 0, 3}, {0, 0, 3}, {3, 3, 3} };
            WHEN("We calculate the angle between these points") {
                const Mat_<double> expected = (Mat_<double>(1, 4) << 1.57079633, 1.57079633, 1.57079633, 0.95531662);
                const auto result = calculateAngleBetweenVectors(v0, v1);
                THEN("The result should be as expected") {
                    REQUIRE(allClose(result, expected));
                }
            }
        }
    }

    SCENARIO("Test angle between two points when direction is set to false")
    {
        GIVEN("Given a point  [1, -2, 3] and another point [-1, 2, -3]")
        {
            Point3d v0{ 1,-2,3 };
            Point3d v1{ -1,2,-3 };
            WHEN("We calculate the angle between these points and set directed to false") {
                const auto result = calculateAngleBetweenVectors(v0, v1, false);
                THEN("The result should be as close to zero as possible") {
                    INFO(result);
                    REQUIRE(allClose(result, 0));
                }
            }
        }
    }

    
    SCENARIO("Test superimposition of a random matrix")
    {
        GIVEN("Given a random 3 x 10 matrix superimposed unto itself")
        {
            const auto v0 = Matrix<double,Dynamic, Dynamic, RowMajor>::Random(3, 10);
            WHEN("We superimpose this matrix onto itself") {
                const auto result = superImpositionMatrix(v0, v0);
                const Matrix4d expected = Matrix4d::Identity();
                THEN("The result should be as close to the 4 x 4 identity matrix") {
                    INFO(result);
                    INFO(expected);
                    REQUIRE(allCloseEigen(result, expected));
                }
            }
        }
    }  
}