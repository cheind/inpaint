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

TEST_CASE("patch-centered")
{
    // Test with cv::Mat

    cv::Mat img = uniformRandomNoiseImage(21);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    // Entire image
    REQUIRE(cv::norm(img, centeredPatch<PATCH_FAST>(img, 21/2, 21/2, 10)) == 0);
    REQUIRE(centeredPatch<PATCH_FAST>(img, 21/2, 21/2, 10).size() == cv::Size(21, 21));
    REQUIRE(cv::norm(img, centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, 21/2, 21/2, 10)) == 0);
    REQUIRE(centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, 21/2, 21/2, 10).size() == cv::Size(21, 21));

    // Top left 9x9 corner
    REQUIRE(cv::norm(img(cv::Range(0,9), cv::Range(0,9)), centeredPatch<PATCH_FAST>(img, 4, 4, 4)) == 0);
    REQUIRE(centeredPatch<PATCH_FAST>(img, 4, 4, 4).size() == cv::Size(9, 9));
    REQUIRE(cv::norm(img(cv::Range(0,9), cv::Range(0,9)), centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, 4, 4, 4)) == 0);
    REQUIRE(centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, 4, 4, 4).size() == cv::Size(9, 9));

    // Bottom right 3x3 corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 3,img.rows), cv::Range(img.cols - 3, img.cols)), centeredPatch<PATCH_FAST>(img, img.rows - 2, img.cols - 2, 1)) == 0);
    REQUIRE(centeredPatch<PATCH_FAST>(img, img.rows - 2, img.cols - 2, 1).size() == cv::Size(3, 3));
    REQUIRE(cv::norm(img(cv::Range(img.rows - 3,img.rows), cv::Range(img.cols - 3, img.cols)), centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, img.rows - 2, img.cols - 2, 1)) == 0);
    REQUIRE(centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, img.rows - 2, img.cols - 2, 1).size() == cv::Size(3, 3));

    // Clamping

    // Top left 9x9 corner, centered at 0,0
    REQUIRE(cv::norm(img(cv::Range(0,5), cv::Range(0,5)), centeredPatch<PATCH_BOUNDS>(img, 0, 0, 4)) == 0);

    // Bottom right 3x3 corner, centered at bottom right corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 2,img.rows), cv::Range(img.cols - 2, img.cols)), centeredPatch<PATCH_BOUNDS>(img, img.rows - 1, img.cols - 1, 1)) == 0);

    // Test with cv::Mat_

    cv::Mat_<cv::Vec3f> m(100, 100);
    m.setTo(cv::Vec3f(1, 2, 3));
    REQUIRE(centeredPatch<PATCH_FAST>(m, 20, 20, 10).size() == cv::Size(21, 21));
    REQUIRE(centeredPatch<PATCH_FAST>(m, 20, 20, 10).type() == m.type());
    REQUIRE(cv::norm(m(cv::Rect(20, 20, 21, 21)), centeredPatch<PATCH_FAST>(m, 20, 20, 10)) == 0);

    REQUIRE(centeredPatch<PATCH_BOUNDS | PATCH_REF>(m, 20, 20, 10).size() == cv::Size(21, 21));
    REQUIRE(centeredPatch<PATCH_BOUNDS | PATCH_REF>(m, 20, 20, 10).type() == m.type());
    REQUIRE(cv::norm(m(cv::Rect(20, 20, 21, 21)), centeredPatch<PATCH_BOUNDS | PATCH_REF>(m, 20, 20, 10)) == 0);
   
}

TEST_CASE("patch-topleft")
{
    // Test with cv::Mat

    cv::Mat img = uniformRandomNoiseImage(21);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    // Entire image
    REQUIRE(cv::norm(img, topLeftPatch<PATCH_FAST>(img, 0, 0, 21, 21)) == 0);
    REQUIRE(topLeftPatch<PATCH_FAST>(img, 0, 0, 21, 21).size() == cv::Size(21, 21));
    REQUIRE(cv::norm(img, topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, 0, 0, 21, 21)) == 0);
    REQUIRE(topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, 0, 0, 21, 21).size() == cv::Size(21, 21));

    // Top left 9x9 corner
    REQUIRE(cv::norm(img(cv::Range(0,9), cv::Range(0,9)), topLeftPatch<PATCH_FAST>(img, 0, 0, 9, 9)) == 0);
    REQUIRE(topLeftPatch<PATCH_FAST>(img, 0, 0, 9, 9).size() == cv::Size(9, 9));
    REQUIRE(cv::norm(img(cv::Range(0,9), cv::Range(0,9)), topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, 0, 0, 9, 9)) == 0);
    REQUIRE(topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, 0, 0, 9, 9).size() == cv::Size(9, 9));

    // Bottom right 3x3 corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 3,img.rows), cv::Range(img.cols - 3, img.cols)), topLeftPatch<PATCH_FAST>(img, img.rows - 3, img.cols - 3, 3, 3)) == 0);
    REQUIRE(topLeftPatch<PATCH_FAST>(img, img.rows - 3, img.cols - 3, 3, 3).size() == cv::Size(3, 3));

    REQUIRE(topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, img.rows - 3, img.cols - 3, 3, 3).size() == cv::Size(3, 3));
    REQUIRE(cv::norm(img(cv::Range(img.rows - 3,img.rows), cv::Range(img.cols - 3, img.cols)), topLeftPatch<PATCH_BOUNDS | PATCH_REF>(img, img.rows - 3, img.cols - 3, 3, 3)) == 0); 

    // Clamping

    // Top left 9x9 corner, anchored at -4, -4
    REQUIRE(cv::norm(img(cv::Range(0,5), cv::Range(0,5)), topLeftPatch<PATCH_BOUNDS>(img, -4, -4, 9, 9)) == 0);

    // Bottom right 3x3 corner, centered at bottom right corner
    REQUIRE(cv::norm(img(cv::Range(img.rows - 2,img.rows), cv::Range(img.cols - 2, img.cols)), topLeftPatch<PATCH_BOUNDS>(img, img.rows - 2, img.cols - 2, 5, 5)) == 0);
}

TEST_CASE("patch-comparable-regions")
{
    cv::Mat a = uniformRandomNoiseImage(40);
    cv::Mat b = uniformRandomNoiseImage(20);

    std::pair<cv::Rect, cv::Rect> p = comparablePatchRegions(a, b, cv::Point(10, 10), cv::Point(10, 10), 2);
    REQUIRE(p.first.size() == p.second.size());
    REQUIRE(p.first.size() == cv::Size(5,5));
    REQUIRE(p.first.tl() == cv::Point(8, 8));
    REQUIRE(p.second.tl() == cv::Point(8, 8));

    p = comparablePatchRegions(a, b, cv::Point(0, 0), cv::Point(10, 10), 2);
    REQUIRE(p.first.size() == p.second.size());
    REQUIRE(p.first.size() == cv::Size(3,3));
    REQUIRE(p.first.tl() == cv::Point(0, 0));
    REQUIRE(p.second.tl() == cv::Point(10, 10));

    p = comparablePatchRegions(a, b, cv::Point(10, 10), cv::Point(0, 0), 2);
    REQUIRE(p.first.size() == p.second.size());
    REQUIRE(p.first.size() == cv::Size(3,3));
    REQUIRE(p.first.tl() == cv::Point(10, 10));
    REQUIRE(p.second.tl() == cv::Point(0, 0));

    p = comparablePatchRegions(a, b, cv::Point(0, 10), cv::Point(10, 10), 2);
    REQUIRE(p.first.size() == p.second.size());
    REQUIRE(p.first.size() == cv::Size(3,5));
    REQUIRE(p.first.tl() == cv::Point(0, 8));
    REQUIRE(p.second.tl() == cv::Point(10, 8));

    p = comparablePatchRegions(a, b, cv::Point(10, 10), cv::Point(19, 19), 2);
    REQUIRE(p.first.size() == p.second.size());
    REQUIRE(p.first.size() == cv::Size(3,3));
    REQUIRE(p.first.tl() == cv::Point(8, 8));
    REQUIRE(p.second.tl() == cv::Point(17, 17));
}