/**
   This file is part of Inpaint.

   Copyright Christoph Heindl 2014

   Inpaint is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   Inpaint is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with Inpaint.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "catch.hpp"
#include "random_testdata.h"

#include <inpaint/mean_shift.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("mean-shift")
{
    cv::Mat_<float> centers(3,2);
    centers << 10, 10, -10, -10, 10, -10;

    cv::Mat_<float> features;
    cv::Mat_<int> labels;
    randomGaussianBlobs(3, 20, 2, 0.4f, centers, features, labels, -10.f, 10.f);

    cv::Mat_<float> msCenters;
    cv::Mat_<int> msLabels;

    // We use weights here just for the purpose implicitly sort the cluster results
    // the same way we provide the inputs.
    cv::Mat_<float> msWeights(1, features.rows);
    msWeights.colRange(0, 20).setTo(2);
    msWeights.colRange(20, 40).setTo(1);
    msWeights.colRange(40, 60).setTo(0.5);

    // Note, 300 iterations are way too much, 5-10 suffice. Tests binning as well.
    meanShift(features, cv::noArray(), msWeights, msCenters, msLabels, cv::noArray(), 1.2f, 300);

    REQUIRE(msCenters.rows == 3);
    REQUIRE(cv::norm(centers, msCenters) < 1);
    REQUIRE(cv::countNonZero(msLabels == 0) == 20);
    REQUIRE(cv::countNonZero(msLabels == 1) == 20);
    REQUIRE(cv::countNonZero(msLabels == 2) == 20);

    // Run again, but provide just one center;
    cv::Mat_<float> oneCenter(1, 2);
    oneCenter << 12, 12;
    meanShift(features, oneCenter, msWeights, msCenters, msLabels, cv::noArray(), 3.0f, 300);

    REQUIRE(msCenters.rows == 1);
    REQUIRE(cv::norm(centers.row(0), msCenters) < 1);

}
