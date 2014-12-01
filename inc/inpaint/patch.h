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

#include <inpaint/stats.h>
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
        Returns a patch anchored on the given top-left corner.

        \tparam Flags Combination of flags for patch creation.

        \param m Underlying image
        \param y y-coordinate of the patch top-left corner
        \param x x-coordinate of the patch top-left corner
        \param height height of patch (extension along y-axis)
        \param width width of patch (extension along x-axis)
        \return Returns a view on the image that contains only the patch region.
    */
    template<int Flags>
    cv::Mat topLeftPatch(const cv::Mat &m, int y, int x, int height, int width) 
    {
        // Note, compile time if's, will be optimized away by compiler.
        if (Flags & PATCH_BOUNDS) {
            int topx = clamp(x, 0, m.cols - 1);
            int topy = clamp(y, 0, m.rows - 1);
            width -= std::abs(topx - x);
            height -= std::abs(topy - y);

	        width = clamp(width, 0, m.cols - topx);
            height = clamp(height, 0, m.rows - topy);            
            x = topx;
            y = topy;
        }

        if (Flags & PATCH_REF) {
            return m(cv::Rect(x, y, width, height));
        } else {
            uchar *start = const_cast<uchar*>(m.ptr<uchar>(y, x));
            return cv::Mat(height, width, m.type(), start, m.step);        
        }
    }

    /** 
        Returns a patch anchored on the given top-left corner.. 
    */
    inline cv::Mat topLeftPatch(const cv::Mat &m, int y, int x, int height, int width) 
    {
        return topLeftPatch<PATCH_FAST>(m, y, x, height, width);
    }

    /** 
        Returns a patch anchored on the given top-left corner.. 
    */
    inline cv::Mat topLeftPatch(const cv::Mat &m, const cv::Rect &r) 
    {
        return topLeftPatch<PATCH_FAST>(m, r.y, r.x, r.height, r.width);
    }
  
    /** 
        Returns a patch centered around the given pixel coordinates.

        \tparam Flags Combination of flags for patch creation.

        \param m Underlying image
        \param y y-coordinate of the patch center
        \param x x-coordinate of the patch center
        \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Returns a view on the image that contains only the patch region.
    */
    template<int Flags>
    cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize) 
    {
        int width = 2 * halfPatchSize + 1;
        int height = 2 * halfPatchSize + 1;
        x -= halfPatchSize;
        y -= halfPatchSize;

        return topLeftPatch<Flags>(m, y, x, height, width);
    }

    /** 
        Returns a patch centered around the given pixel coordinates. 
    */
    inline cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize) 
    {
        return centeredPatch<PATCH_FAST>(m, y, x, halfPatchSize);
    }

    /** 
        Given two centered patches in two images compute the comparable region in both images as top-left patches. 
        
        \param a first image
        \param b second image
        \param ap center in first image
        \param bp center in second image
        \param halfPatchSize halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
        \return Comparable rectangles for first, second image. Rectangles are of same size, but anchored top-left
                with respect to the given center points.
        */
    inline std::pair<cv::Rect, cv::Rect> comparablePatchRegions(
        const cv::Mat &a, const cv::Mat &b,
        cv::Point ap, cv::Point bp, 
        int halfPatchSize)
    {
        int left = maximum(-halfPatchSize, -ap.x, -bp.x);
        int right = minimum(halfPatchSize + 1, -ap.x + a.cols, -bp.x + b.cols); 
        int top = maximum(-halfPatchSize, -ap.y, -bp.y);
        int bottom = minimum(halfPatchSize + 1, -ap.y + a.rows, -bp.y + b.rows); 

        std::pair<cv::Rect, cv::Rect> p;

        p.first.x = ap.x + left;
        p.first.y = ap.y + top;
        p.first.width = (right - left);
        p.first.height = (bottom - top);

        p.second.x = bp.x + left;
        p.second.y = bp.y + top;
        p.second.width = (right - left);
        p.second.height = (bottom - top);

        return p;
    }

    /** Test if patch goes across the boundary. */
    inline bool isCenteredPatchCrossingBoundary(cv::Point p, int halfPatchSize, const cv::Mat &img)
    {
        return p.x < halfPatchSize || p.x >= img.cols - halfPatchSize ||
               p.y < halfPatchSize || p.y >= img.rows - halfPatchSize;
    }
    

}
#endif