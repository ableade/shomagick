#include "stdafx.h"
#include "catch.hpp"
#include "../src/allclose.h"
#include <Eigen/Core>

#include <vector>
#include <ostream>
#include <iostream>
#include <string>

using std::vector;
using std::cerr;

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

namespace catchtests
{
    SCENARIO("two collections with close elements should be all close", "[allClose]")
    {
        GIVEN("two collections that are close and a tolerance that we set manually")
        {
            using std::vector;
            const auto collection1 = vector<float>{
                1, 2, 3
            };

            const auto collection2 = vector<float>{
                1.001, 2, 3
            };

            constexpr auto tolerance = 0.05;

            WHEN("the collections are checked for closeness")
            {
                const auto actual = allClose(
                    collection1,
                    collection2,
                    tolerance
                );

                THEN("the results shoud be as expected")
                {
                    constexpr auto expected = true;
                    INFO("collection1: " << makePrintable(collection1));
                    INFO("collection2: " << makePrintable(collection2));

                    cerr << "expected = " << expected << ", actual = " << actual << "\n";
                    
                    REQUIRE( expected == actual );
                }
            }
        }
    }

    SCENARIO("two Eigen::Vector3d with close elements should be all close", "[allClose]")
    {
        GIVEN("two Eigen::Vector3d  that are close and a tolerance that we set manually")
        {
            using std::vector;
            const auto collection1 = Eigen::Vector3d{
                1, 2, 3
            };

            const auto collection2 = Eigen::Vector3d{
                1.001, 2, 3
            };

            constexpr auto tolerance = 0.05;

            WHEN("the collections are checked for closeness")
            {
                const auto actual = allClose(
                    collection1,
                    collection2,
                    tolerance
                );

                THEN("the results shoud be as expected")
                {
                    const auto expected = true;
                    INFO("collection1: " << collection1);
                    INFO("collection2: " << collection2);
                    REQUIRE(expected == actual);
                }
            }
        }
    }
} //namespace catchtests
