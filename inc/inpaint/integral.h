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

#ifndef INPAINT_INTEGRAL_H
#define INPAINT_INTEGRAL_H

#include <opencv2/core/core.hpp>

namespace Inpaint {

    /** 
        Compute the sum of elements in the given rectangle using an integral image.
        No bounds checking is performed.

        Each image channel is evaluated seperately. Currently the only valid channel 
        type is CV_32S.

        \param i Integral image
        \param r Rectangle to compute sum for
        \return sum as scalar. 
    */
    inline cv::Scalar sumInRectUsingIntegralImage(const cv::Mat &i, const cv::Rect &r)
    {
        const cv::Point_<int> tl = r.tl();
        const cv::Point_<int> br = r.br(); // br is exclusive. Ok as integral image is larger by one.

        const int *tRow = i.ptr<int>(r.tl().y);
        const int *bRow = i.ptr<int>(r.br().y);

        cv::Scalar sum(0);
        const int n = i.channels();

        for (int i = 0; i < n; ++i) {
            sum[i] = bRow[n * br.x + i] - bRow[n * tl.x + i] - tRow[n * br.x + i] + tRow[n * tl.x + i];
        }

        return sum;
    }

}
#endif