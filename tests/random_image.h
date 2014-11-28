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

#ifndef INPAINT_RANDOM_IMAGE_H
#define INPAINT_RANDOM_IMAGE_H

#include <opencv2/opencv.hpp>

inline cv::Mat randomLinesImage(int imageSize, int nLines)
{
    cv::Mat m(imageSize, imageSize, CV_8UC1);
    m.setTo(0);

    cv::RNG rng(10);
    cv::Point pt1, pt2;
    for( int i = 0; i < nLines; ++i) {
        pt1.x = rng.uniform( 0, imageSize );
        pt1.y = rng.uniform( 0, imageSize );
        pt2.x = rng.uniform( 0, imageSize );
        pt2.y = rng.uniform( 0, imageSize );
 
        cv::line(m, pt1, pt2, rng.uniform(10, 255), rng.uniform(1, 10));        
    }

    return m;
}

inline cv::Mat uniformRandomNoiseImage(int imageSize)
{
    cv::Mat m(imageSize, imageSize, CV_8UC1);
    cv::RNG rng(10);
    for( int i = 0; i < imageSize * imageSize; ++i) {
        m.at<uchar>(i) = rng.uniform(0,255);
    }

    return m;
}

inline cv::Mat randomBlock(const cv::Mat &image, cv::Rect &r)
{
    cv::RNG rng;

    int x1 = rng.uniform(0, image.cols - 1);
    int x2 = rng.uniform(0, image.cols - 1);
    
    if (x2 < x1) {
        std::swap(x1, x2);
    }

    int y1 = rng.uniform(0, image.rows - 1);
    int y2 = rng.uniform(0, image.rows - 1);
    
    if (y2 < y1) {
        std::swap(y1, y2);
    }

    r = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    return image(r);
}
    

#endif