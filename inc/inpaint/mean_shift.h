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

#ifndef INPAINT_MEAN_SHIFT_H
#define INPAINT_MEAN_SHIFT_H

#include <opencv2/core/core.hpp>

namespace Inpaint {
    
    /** 
        Non-parametric clustering using mean-shift and a flat kernel.

        Implementation is based on
        "Mean Shift: A Robust Approach toward Feature Space Analysis" D. Comaniciu et al.

        Changes made by the author with respect to the original paper:
            - in order to avoid local maxima (saddle points), converged cluster centers
              are perturbated by a fraction of the bandwidth. When the point converges
              again to the same cluster center, the center value is said to be final.

            - if no seeds are provided, the method bins the features in a grid of grid-size
              equal to bandwidth. Seeds will then be placed in bins with at least on element
              in them.

        \param features Matrix of features. One feature per row. Required depth: CV_32F.
        \param seeds Optional matrix of seeds to use. One seed per row. Required depth: CV_32F. If omitted, binning is used
                     to generate seeds.
        \param weights Optional matrix of weights to apply to features. Required type: CV_32FC1 of size 1 x number of features.
        \param centers Cluster centers. Depth: CV_32F.
        \param labels Feature-to-cluster lables if required. Type: CV_32SC1.
        \param distances Feature-to-cluster distances based on L2 norm if required. Type: CV_32FC1.
        \param bandwidth Fixed radius during iteration.
        \param maxIteration No more iterations per seed will be performed.
        \param perturbate When seeds converge, perturbate them to see if they converge back to the same spot.
        */      void meanShift(
        cv::InputArray features, cv::InputArray seeds, cv::InputArray weights, 
        cv::OutputArray centers, cv::OutputArray labels, cv::OutputArray distances,     
        float bandwidth, int maxIteration = 200, bool perturbate = true);

   

}

#endif