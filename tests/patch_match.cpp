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

#include <inpaint/patch_match.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("patch-match")
{
    cv::Mat img = randomLinesImage(200, 60);
    cv::Mat shifted = shiftImage(img, 15, 20);

    cv::Mat corrs, distances;
    patchMatch(img, shifted, cv::noArray(), corrs, distances, 5, 5);

    cv::Mat reconstructed(img.size(), CV_8UC1);
    for (int y = 0; y < reconstructed.rows; ++y) {
        for (int x = 0; x < reconstructed.cols; ++x) {
            cv::Vec2i corr = corrs.at<cv::Vec2i>(y, x);
            reconstructed.at<uchar>(y,x) = shifted.at<uchar>(corr[1], corr[0]);
        }
    }

    cv::Rect r(0, 0, img.cols - 30, img.rows - 30);
    REQUIRE((cv::norm(img(r), reconstructed(r), cv::NORM_L1) / r.area()) < 5);
}

TEST_CASE("patch-match-mask")
{
    cv::Mat img = randomLinesImage(200, 60);
    
    cv::Mat mask(img.size(), CV_8UC1);
    mask.setTo(255);
    cv::Rect r(20, 20, 80, 80);
    cv::rectangle(mask, r, cv::Scalar(0), cv::FILLED);

    cv::Mat corrs, distances;
    patchMatch(img, img, mask, corrs, distances, 5, 5);

    bool noneInside = true;
    cv::Vec2i wrong;
    for (int i = 0; i < img.rows * img.cols; ++i) {
        cv::Vec2i corr = corrs.at<cv::Vec2i>(i);
        if (r.contains(corr)) {
            noneInside = false;
            wrong = corr;
            break;
        }            
    }    
    REQUIRE(noneInside);

}

TEST_CASE("patch-match-subwindow-as-source")
{
    cv::Mat img = randomLinesImage(200, 60);
    cv::Mat subimg = centeredPatch(img, 40, 40, 30);

    cv::Mat corrs, distances;
    patchMatch(subimg, img, cv::noArray(), corrs, distances, 10, 10);

    cv::Mat reconstructed(subimg.size(), CV_8UC1);
    for (int y = 0; y < reconstructed.rows; ++y) {
        for (int x = 0; x < reconstructed.cols; ++x) {
            cv::Vec2i corr = corrs.at<cv::Vec2i>(y, x);
            reconstructed.at<uchar>(y,x) = img.at<uchar>(corr[1], corr[0]);
        }
    }

    // Borders are incorrect because less pixels to test.
    REQUIRE(cv::norm(subimg(cv::Rect(10, 10, 50, 50)), reconstructed(cv::Rect(10, 10, 50, 50))) < 100);
}
