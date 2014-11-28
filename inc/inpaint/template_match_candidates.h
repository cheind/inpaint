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

#ifndef INPAINT_TEMPLATE_MATCH_CANDIDATES_H
#define INPAINT_TEMPLATE_MATCH_CANDIDATES_H

#include <opencv2/core/core.hpp>

namespace Inpaint {

    /**
        Find candidate positions for template matching.

        Implementation based on 
        "Speed-up Template Matching through Integral Image based Weak Classifiers", Tirui Wu et. al.
    */
    class TemplateMatchCandidates {
    public:
        void setSourceImage(const cv::Mat &image);
        void setTemplateSize(cv::Size templateSize);
        void setPartitionSize(cv::Size s);
        void initialize();

        cv::Mat findCandidates(
            const cv::Mat &templ, 
            const cv::Mat &templMask = cv::Mat(), 
            int maxWeakErrors = 3, 
            float maxMeanDifference = 20);

    private:

        void computeBlockRects(
            cv::Size size, 
            cv::Size partitions, 
            std::vector< cv::Rect > &rects);

        void removeInvalidBlocks(
            const cv::Mat &templMask, 
            std::vector< cv::Rect > &rects);

        void weakClassifiersForTemplate(
            const cv::Mat &templ, 
            const cv::Mat &templMask, 
            const std::vector< cv::Rect > &rects, 
            cv::Mat_<int> &classifiers, 
            cv::Scalar &mean);

        uchar compareWeakClassifiers(
            const cv::Mat_<int> &i, 
            int x, int y, 
            cv::Size templSize, 
            const std::vector< cv::Rect > &blocks, 
            const int *compareTo, 
            float templateMean, 
            float maxMeanDiff, int maxWeakErrors);

        cv::Mat _image;
        std::vector< cv::Mat_<int> > _integrals;
        std::vector< cv::Rect > _blocks;
        cv::Mat_<uchar> _candidates;
        cv::Size _templateSize;
        cv::Size _partitionSize;
    };


    cv::Mat findTemplateMatchCandidates(
        const cv::Mat &image,
        const cv::Mat &templ,
        const cv::Mat &templMask = cv::Mat(),
        cv::Size partitionSize = cv::Size(3,3),
        int maxWeakErrors = 3, 
        float maxMeanDifference = 20);

}
#endif