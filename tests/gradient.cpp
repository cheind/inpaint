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

#define CATCH_CONFIG_MAIN  
#include "catch.hpp"
#include "random_image.h"

#include <inpaint/gradient.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("gradient")
{
    cv::Mat img = randomLinesImage(50, 50);

    // Reference
    cv::Mat gradX, gradY;
    cv::Sobel(img, gradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_CONSTANT);
	cv::Sobel(img, gradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_CONSTANT);

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            const cv::Vec2f gref(gradX.at<float>(cv::Point(x,y)), gradY.at<float>(cv::Point(x,y)));
            
            cv::Vec2f g = gradient(img, y, x);            
            REQUIRE(g[0] == Approx(gref[0]));
            REQUIRE(g[1] == Approx(gref[1]));

            cv::Vec2f gn = normalizedGradient(img, y, x);
            cv::Vec2f gnref = gref;
            
            float d = gref.dot(gref);
            if (d == 0) {
                gnref *= 0;
            } else {
                gnref /= sqrtf(d);
            }

            REQUIRE(gn[0] == Approx(gnref[0]));
            REQUIRE(gn[1] == Approx(gnref[1]));
        }
    }
}
