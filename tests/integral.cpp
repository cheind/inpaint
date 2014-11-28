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

#include <inpaint/integral.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("integral")
{
    cv::Mat img(4, 4, CV_8UC3);
    img.setTo(cv::Scalar(1, 2, 3));

    cv::Mat i;
    cv::integral(img, i);

    REQUIRE(sumInRectUsingIntegralImage(i, cv::Rect(0, 0, 2, 2)) == cv::Scalar(1*4, 2*4, 3*4));
    REQUIRE(sumInRectUsingIntegralImage(i, cv::Rect(1, 1, 3, 3)) == cv::Scalar(1*9, 2*9, 3*9));
}