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

#include <inpaint/patch.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("patch-no-bounds")
{
    cv::Mat img = uniformRandomNoiseImage(21);

    // Entire image
    REQUIRE(cv::norm(img, centeredPatch(img, 21/2, 21/2, 10)) == 0);
    REQUIRE(centeredPatch(img, 21/2, 21/2, 10).size() == cv::Size(21, 21));

    // Top left 9x9 corner
    REQUIRE(cv::norm(img(cv::Range(0,9), cv::Range(0,9)), centeredPatch(img, 4, 4, 4)) == 0);
    REQUIRE(centeredPatch(img, 4, 4, 4).size() == cv::Size(9, 9));

    // Bottom right 3x3 corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 3,img.rows), cv::Range(img.cols - 3, img.cols)), centeredPatch(img, img.rows - 2, img.cols - 2, 1)) == 0);
    REQUIRE(centeredPatch(img, img.rows - 2, img.cols - 2, 1).size() == cv::Size(3, 3));
}

TEST_CASE("patch-clamp")
{
    cv::Mat img = uniformRandomNoiseImage(21);

    // Entire image
    REQUIRE(cv::norm(img, centeredPatchClamped(img, 21/2, 21/2, 10)) == 0);

    // Top left 9x9 corner, centered at 0,0
    REQUIRE(cv::norm(img(cv::Range(0,5), cv::Range(0,5)), centeredPatchClamped(img, 0, 0, 4)) == 0);

    // Bottom right 3x3 corner, centered at bottom right corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 2,img.rows), cv::Range(img.cols - 2, img.cols)), centeredPatchClamped(img, img.rows - 1, img.cols - 1, 1)) == 0);
}