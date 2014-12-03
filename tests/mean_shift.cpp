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

#include "catch.hpp"
#include "random_testdata.h"

#include <inpaint/mean_shift.h>
#include <opencv2/opencv.hpp>

using namespace Inpaint;

TEST_CASE("mean-shift")
{
   cv::Mat_<float> centers(3,2);
   centers << 10, 10, -10, -10, 10, -10;

   cv::Mat_<float> features;
   cv::Mat_<int> labels;

   randomGaussianBlobs(3, 100, 2, 0.4f, centers, features, labels, -10.f, 10.f);

   std::cout << "init ok" << std::endl;
   cv::Mat_<float> msCenters;
   cv::Mat_<int> msLabels;
   meanShift(features, cv::noArray(), cv::noArray(), msCenters, msLabels, cv::noArray(), 1.2f, 300);

   /*
   ms = MeanShift(bandwidth=bandwidth)
    labels = ms.fit(X).labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    assert_equal(n_clusters_, n_clusters)

    cluster_centers, labels = mean_shift(X, bandwidth=bandwidth)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    assert_equal(n_clusters_, n_clusters)*/
}