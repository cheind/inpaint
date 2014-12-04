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
        
        Besides clustering this method can also be used for robust mean estimation.

        Implementation is based on
        "Mean Shift: A Robust Approach toward Feature Space Analysis" D. Comaniciu et al.

        Changes made by the author with respect to the original paper:
            - in order to avoid local maxima (saddle points), potentially converged cluster centers
              are perturbated by a fraction of the bandwidth. When the point converges
              again to the same cluster center, the cluster center to be converged.

            - if no seeds are provided, the method bins the features in a grid of grid-size
              equal to bandwidth. Seeds will then be placed in bins with at least on element
              in them.

        \param features Matrix of features of size num-features x num-dims and type CV_32FC1.     
        \param seeds Optional matrix of seeds of size num-seeds x num-dims and type CV_32FC1. 
               If omitted, binning is used to generate seed points.
        \param weights Optional matrix of feature weights of size 1 x num-features and type CV_32FC1.
        \param centers Cluster means matrix of size num-clusters-found x num-dims and type CV_32FC1.
        \param labels Optionally computed feature-to-cluster label matrix of size 1 x num-features and type CV_32SC1.
        \param distances Optionally computed feature-to-cluster squared L2 distance matrix of size 1 x num-features and 
               type CV_32FC1.
        \param bandwidth Fixed radius during iteration.
        \param maxIteration No more iterations per seed will be performed.
        \param perturbate When seeds converge, perturbate them to see if they converge back to the same spot.
        \param mergeClusters Merge all cluster centers that are closer than bandwidth.
        \param sortClusters Sort clusters descending based on probability modes in descending order.

        */      void meanShift(
        cv::InputArray features, cv::InputArray seeds, cv::InputArray weights, 
        cv::OutputArray centers, cv::OutputArray labels, cv::OutputArray distances,     
        float bandwidth, int maxIteration = 200, bool perturbate = true, bool mergeClusters = true, bool sortClusters = true);

   

}

#endif