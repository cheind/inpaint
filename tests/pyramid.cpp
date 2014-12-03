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
#include "random_image.h"

#include <inpaint/pyramid.h>

using namespace Inpaint;

TEST_CASE("pyramid")
{
    cv::Mat imgColor(240, 320, CV_8UC3);
    cv::Mat imgInts(240, 320, CV_32SC2);
    cv::Mat imgDoubles(240, 320, CV_64FC1);

    std::vector<cv::Mat> pyr;
    imagePyramid(imgColor, pyr, cv::Size(40, 30), cv::INTER_NEAREST);
    REQUIRE(pyr.size() == 4);
    REQUIRE(pyr[0].size() == cv::Size(320, 240));
    REQUIRE(pyr[1].size() == cv::Size(160, 120));
    REQUIRE(pyr[2].size() == cv::Size(80, 60));
    REQUIRE(pyr[3].size() == cv::Size(40, 30));

    imagePyramid(imgInts, pyr, cv::Size(40, 30), cv::INTER_NEAREST);
    REQUIRE(pyr.size() == 4);
    REQUIRE(pyr[0].size() == cv::Size(320, 240));
    REQUIRE(pyr[1].size() == cv::Size(160, 120));
    REQUIRE(pyr[2].size() == cv::Size(80, 60));
    REQUIRE(pyr[3].size() == cv::Size(40, 30));

    imagePyramid(imgDoubles, pyr, cv::Size(40, 30), cv::INTER_NEAREST);
    REQUIRE(pyr.size() == 4);
    REQUIRE(pyr[0].size() == cv::Size(320, 240));
    REQUIRE(pyr[1].size() == cv::Size(160, 120));
    REQUIRE(pyr[2].size() == cv::Size(80, 60));
    REQUIRE(pyr[3].size() == cv::Size(40, 30));
}

