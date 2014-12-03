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

#ifndef INPAINT_PATCH_MATCH_H
#define INPAINT_PATCH_MATCH_H

#include <inpaint/patch.h>
#include <inpaint/timer.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace Inpaint {

    /**
        Compute dense approximate nearest neighbor fields.

        Implementation is based on 
        "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing", Barnes et al.

        Iterative method to compute the nearest neighbor in target for each pixel in source. Comparison
        for best pixels can use any of the supported cv::norm metric types.

        \param source image. Either 1 channel or 3 channel images are supported.
        \param target image. Target image 

    */
    void patchMatch(
        cv::InputArray &source, 
        cv::InputArray &target, cv::InputArray &targetMask, 
        cv::InputOutputArray &corrs, cv::InputOutputArray &distances,
        int halfPatchSize,
        int iterations,
        int normType = cv::NORM_L2SQR);


    

}
#endif