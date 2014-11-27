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

#ifndef INPAINT_PATCH_H
#define INPAINT_PATCH_H

#include <opencv2/core/core.hpp>

namespace Inpaint {

    /** 
        Returns a patch centered around the given pixel coordinates.
        No bounds checking is performed.

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<class Mat>
    Mat centeredPatch(const Mat &m, int y, int x, int halfPatchSize) 
    {
        return m(cv::Rect(x - halfPatchSize, y - halfPatchSize, halfPatchSize * 2 + 1, halfPatchSize * 2 + 1));
    }

    /** 
        Returns a patch centered around the given pixel coordinates.
        No bounds checking is performed.

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<class Mat>
    Mat centeredPatch(const Mat &m, cv::Point p, int halfPatchSize) 
    {
        return centeredPatch(m, p.y, p.x, halfPatchSize);
    }

    /** 
        Returns a patch centered around the given pixel coordinates.
        Clamps the patch size where it would otherwise reach out of image bounds.

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<class Mat>
    Mat centeredPatchClamped(const Mat &m, int y, int x, int halfPatchSize) 
    {
        const int topx = std::max<int>(x - halfPatchSize, 0);
	    const int topy = std::max<int>(y - halfPatchSize, 0);
	    const int bottomx = std::min<int>(x + halfPatchSize, m.cols - 1);
	    const int bottomy = std::min<int>(y + halfPatchSize, m.rows - 1);

	    return m(cv::Rect(topx, topy, bottomx - topx + 1, bottomy - topy + 1));
    }

    /** 
        Returns a patch centered around the given pixel coordinates.
        Clamps the patch size where it would otherwise reach out of image bounds.

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<class Mat>
    Mat centeredPatchClamped(const Mat &m, cv::Point p, int halfPatchSize) 
    {
        return centeredPatchClamped(m, p.y, p.x, halfPatchSize);
    }

}
#endif