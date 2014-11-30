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

    /** Flags for creating patch. */
    enum PatchFlags {
        /** No flags. Fastest variant. */
        PATCH_FAST = 0,
        /** Clamp patch to bounds of image. */
        PATCH_BOUNDS = 1 << 1,
        /** Reference parent memory. Slower, but keeps the parent memory alive. */
        PATCH_REF = 1 << 2
    };
  
    /** 
        Returns a patch centered around the given pixel coordinates.

        \tparam Flags Combination of flags for patch creation.
        \tparam T Data-type of cv::Mat_

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<int Flags, class T>
    cv::Mat_<T> centeredPatch(const cv::Mat_<T> &m, int y, int x, int halfPatchSize) 
    {
        int width, height;

        // Note, if's are compile time if's and will be optimized away.
        if (Flags & PATCH_BOUNDS) {
            const int topx = std::max<int>(x - halfPatchSize, 0);
	        const int topy = std::max<int>(y - halfPatchSize, 0);
	        const int bottomx = std::min<int>(x + halfPatchSize, m.cols - 1);
            const int bottomy = std::min<int>(y + halfPatchSize, m.rows - 1);

            x = topx;
            y = topy;
            width = bottomx - topx + 1;
            height = bottomy - topy + 1;

        } else {           
            x -= halfPatchSize;
            y -= halfPatchSize;
            width = halfPatchSize * 2 + 1;
            height = halfPatchSize * 2 + 1;
        }

        if (Flags & PATCH_REF) {
            return m(cv::Rect(x, y, width, height));
        } else {
            T *start = const_cast<T*>(m.ptr<T>(y, x));
            return cv::Mat_<T>(height, width, start, m.step);        
        }
    }

    /** 
        Returns a patch centered around the given pixel coordinates.

        \tparam Flags Combination of flags for patch creation.
        \tparam T Data-type of cv::Mat_

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<int Flags>
    cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize) 
    {
        int width, height;

        if (Flags & PATCH_BOUNDS) {
            const int topx = std::max<int>(x - halfPatchSize, 0);
	        const int topy = std::max<int>(y - halfPatchSize, 0);
	        const int bottomx = std::min<int>(x + halfPatchSize, m.cols - 1);
            const int bottomy = std::min<int>(y + halfPatchSize, m.rows - 1);

            x = topx;
            y = topy;
            width = bottomx - topx + 1;
            height = bottomy - topy + 1;

        } else {           
            x -= halfPatchSize;
            y -= halfPatchSize;
            width = halfPatchSize * 2 + 1;
            height = halfPatchSize * 2 + 1;
        }

        if (Flags & PATCH_REF) {
            return m(cv::Rect(x, y, width, height));
        } else {
            uchar *start = const_cast<uchar*>(m.ptr<uchar>(y, x));
            return cv::Mat(height, width, m.type(), start, m.step);        
        }
    }

    /** 
        Returns a patch centered around the given pixel coordinates. 
    */
    inline cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize) 
    {
        return centeredPatch<PATCH_FAST>(m, y, x, halfPatchSize);
    }

    /** 
        Returns a patch centered around the given pixel coordinates.
    */
    template <class T>
    cv::Mat_<T> centeredPatch(const cv::Mat_<T> &m, int y, int x, int halfPatchSize) 
    {
        return centeredPatch<PATCH_FAST>(m, y, x, halfPatchSize);
    }



    

}
#endif