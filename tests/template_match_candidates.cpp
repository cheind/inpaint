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

#include <inpaint/template_match_candidates.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("template-match-candidates")
{
    cv::Mat img = randomLinesImage(200, 40);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    cv::Rect r;
    cv::Mat block = randomBlock(img, r);

    cv::Mat candidates;
    findTemplateMatchCandidates(img, block, cv::Mat(), candidates, cv::Size(4,4), 0);
    REQUIRE(candidates.at<uchar>(r.tl()) != 0);
    REQUIRE(cv::countNonZero(candidates) < 20);
}
