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

#ifndef INPAINT_GRADIENT_H
#define INPAINT_GRADIENT_H

#include <opencv2/core/core.hpp>

namespace Inpaint {

    /** 
        Compute the gradient at the given point using the Sobel operator.

        Only intended for sparse computations. For dense gradients refer to
        cv::Sobel. No bounds checking is done and it assumes single channel
        images.

        \param m Image to operate on.
        \param y y-coordinate of target point.
        \param x x-coordinate of target point.
        \return Returns the gradient.

      */
    template<class Mat>
    cv::Vec2f gradient(const Mat &m, int y, int x)
    {
        const uchar *rows[3] = {
            m.ptr<uchar>(y-1),
            m.ptr<uchar>(y),
            m.ptr<uchar>(y+1)
        };

        const float gx = -1.f * rows[0][x-1] + 1.f * rows[0][x+1] + 
                         -2.f * rows[1][x-1] + 2.f * rows[1][x+1] + 
                         -1.f * rows[2][x-1] + 1.f * rows[2][x+1];

        const float gy = -1.f * rows[0][x-1] + -2.f * rows[0][x] + -1.f * rows[0][x+1] +
                          1.f * rows[2][x-1] +  2.f * rows[2][x] +  1.f * rows[2][x+1];

        return cv::Vec2f(gx, gy);
    }

    /** 
        Compute the normalized gradient at the given point. 

        \param m Image to operate on.
        \param y y-coordinate of target point.
        \param x x-coordinate of target point.
        \return Returns the normalized gradient.

     */
    template<class Mat>
    cv::Vec2f normalizedGradient(const Mat &m, int y, int x)
    {
        cv::Vec2f grad = gradient(m, y, x);
		float dot = grad.dot(grad);

		if (dot == 0) {
			grad *= 0;
		} else {
			grad /= sqrtf(dot);
		}

        return grad;
    }


}
#endif