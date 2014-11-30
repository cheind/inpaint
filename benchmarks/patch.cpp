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
#include <inpaint/patch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace Inpaint;

TEST_CASE("patch")
{
    cv::Mat img(100, 100, CV_8UC3);
    const int niter = 500000;

    Timer t;
    int sum = 0;
    for (int i = 0; i < niter; ++i) {
        sum += centeredPatch<PATCH_BOUNDS | PATCH_REF>(img, 10, 10, 10).rows;
    }
    std::cout << "centeredPatch<PATCH_BOUNDS | PATCH_REF>:  " << ((t.measure() * 1000))<< " msec."  << std::endl;

    sum = 0;
    for (int i = 0; i < niter; ++i) {
        sum += centeredPatch<PATCH_BOUNDS>(img, 10, 10, 10).rows;
    }
    std::cout << "centeredPatch<PATCH_BOUNDS>:  " << ((t.measure() * 1000))<< " msec."  << std::endl;
    
    sum = 0;
    for (int i = 0; i < niter; ++i) {
        sum += centeredPatch<PATCH_REF>(img, 10, 10, 10).rows;
    }
    std::cout << "centeredPatch<PATCH_REF>:  " << ((t.measure() * 1000))<< " msec."  << std::endl;

    sum = 0;
    for (int i = 0; i < niter; ++i) {
        sum += centeredPatch<PATCH_FAST>(img, 10, 10, 10).rows;
    }
    std::cout << "centeredPatch<PATCH_FAST>:  " << ((t.measure() * 1000))<< " msec."  << std::endl;
}
