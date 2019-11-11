#include <catch.hpp>
#include "multiview.h"
#include "allclose.h"
#include <opencv2/core.hpp>
#include "random_generators.hpp"

using std::vector;
using cv::Vec3d;
using cv::Mat;
using cv::Matx33d;
using cv::Mat_;
using opengv::translation_t;
using opengv::rotation_t;
using opengv::bearingVectors_t;
using opengv::points_t;

namespace catchtests
{
    SCENARIO("Test plane estimation with no vectors or verticals")
    {
        GIVEN("A point with no vectors or verticals")
        {
            Mat points = (Mat_<double>(3, 3) << 0, 0, 0,
                1, 1, 0, 
                0,1,0);
            Mat vectors;
            Mat verticals;


            WHEN("We estimate the plane from these parameters") {
                const auto p = fitPlane(points, vectors, verticals);
                THEN("The result should be as expected") {
                    std::cout << "P was " << p << "\n";
                    const ShoColumnVector4d expected { 0,0,1,0 };
                    REQUIRE(allClose(p, expected));
                }
            }
        }
    }

    SCENARIO("Test plane estimation with points and vectors only")
    {
        GIVEN("A point with vectors but no verticals")
        {
            Mat points = (Mat_<double>(2, 3) << 0, 0, 0,
                1,1,0);
            vector<ShoColumnVector3d> vectors{ {1,0,0} };
            Mat verticals;


            WHEN("We estimate the plane from these parameters") {
                const auto p = fitPlane(points, Mat(vectors), verticals);
                THEN("The result should be as expected") {
                    const auto expected = cv::Mat(cv::Vec4d{ 0,0,1,0 });
                    const auto expected2 = cv::Mat(cv::Vec4d{ 0, 0, -1,0 });
                    const auto result = (allClose(p, expected) || allClose(p, expected2));
                    REQUIRE(result);
                }
            }
        }
    }


    SCENARIO("Test plane estimation with points, vectors and verticals")
    {
        GIVEN("A point with vectors and verticals")
        {
            Mat points = (Mat_<double>(2, 3) << 0, 0, 0,
                0, 1, 0);
            std::vector<ShoColumnVector3d> vectors{ {1,0,0} };
            std::vector<ShoColumnVector3d> verticals{ {0,0,1} };


            WHEN("We estimate the plane from these parameters") {
                const auto p = fitPlane(points, Mat(vectors), Mat(verticals));
                THEN("The result should be as expected") {
                    const auto expected = cv::Mat(cv::Vec4d{ 0,0,1,0 });
                    const auto result = (allClose(p, expected));
                    REQUIRE(result);
                }
            }
        }
    }

    SCENARIO("Test get standard deviation across the row axis")
    {
        GIVEN("A matrix with three rows and three columns")
        {
            Matx33d a{1,2,3,4,5,6,7,8,9};
            vector<float> expected{ 2.44948974, 2.44948974, 2.44948974 };
            WHEN("We calculate the standard deviation across the first axis") {
                const auto p = getStdByAxis(a, 0);
                THEN("The result should be as expected") {
                    const auto result = (allClose(p, expected));
                    REQUIRE(result);
                }
            }
        }
    }

    SCENARIO("Test get mean across the row axis")
    {
        GIVEN("A matrix with three rows and three columns")
        {
            Matx33d a{ 1,2,3,4,5,6,7,8,9 };
            vector<float> expected{ 4.0,5.0,6.0};
            WHEN("We calculate the standard deviation across the first axis") {
                const auto p = getMeanByAxis(a, 0);
                THEN("The result should be as expected") {
                    const auto result = (allClose(p, expected));
                    REQUIRE(result);
                }
            }
        }
    }

    SCENARIO("Test resecting a 3d points with 3d points")
    {
        GIVEN("A set of ten 2d and 3d points, noise 0.0, outlierfraction 0.1")
        {
            translation_t position = opengv::generateRandomTranslation(2.0);
            rotation_t rotation = opengv::generateRandomRotation(0.5);

            //create a fake central camera
            translations_t camOffsets;
            rotations_t camRotations;
            generateCentralCameraSystem(camOffsets, camRotations);


            bearingVectors_t bearingVectors;
            points_t points;
            std::vector<int> camCorrespondences; //unused in the central case!
            Eigen::MatrixXd gt(3, numberPoints);
            generateRandom2D3DCorrespondences(
                position, rotation, camOffsets, camRotations, numberPoints, noise, outlierFraction,
                bearingVectors, points, camCorrespondences, gt);

            //print the experiment characteristics
            printExperimentCharacteristics(
                position, rotation, noise, outlierFraction);

            WHEN("We calculate the absolute pose") {
                
                THEN("The result should be as expected") {
           
                }
            }
        }
    }
}