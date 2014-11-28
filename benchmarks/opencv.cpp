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

#include <inpaint/timer.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace Inpaint;

TEST_CASE("opencv-region")
{
    cv::Mat img(100, 100, CV_8UC3);

    Timer t;
    int sum = 0;
    cv::Rect r(10, 10, 20, 20);
    for (int i = 0; i < 500000; ++i) {
        sum += img(r).rows;
    }

    std::cout << "operator() took: " << t.measure() * 1000 << " msec." << std::endl;
    
    sum = 0;
    for (int i = 0; i < 500000; ++i) {
        uchar *start = img.ptr<uchar>(10, 10);
        sum += cv::Mat(20,20,CV_8UC3, start, img.step[0]).rows;
    }

    std::cout << "hand-crafted took: " << t.measure() * 1000 << " msec." << std::endl;
}
