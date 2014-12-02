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

        Implementation is based on 
        "Speed-up Template Matching through Integral Image based Weak Classifiers", Tirui Wu et. al.

        This method subdivides the template into a set of blocks. For each block the mean (of each
        channel) is calculated and compared to the mean of the entire template. A binary decision is
        based on whether the block mean is bigger than the template mean. Such a decision is made for
        each block.

        Then for each template position in the source image, the same procedure as described above is
        applied (replace "template" with "region of image") and the binary decisions are compared. If
        too many descisions mismatch, the current position is said not to be a candidate.

        What makes the method fast is the use of integral images for calculating means. Using integral
        images reduces the calculated of the mean to two additions, two subtractions and a division, 
        independent of the size of the rectangle to calculate the mean for.

        Changes made by the author with respect to the original paper:
            - The algorithm works with 1 or 3 channel images. If multiple image channels are present
              each channel is treated seperately.

            - A mask can be passed with the template. The mask defines the valid areas within the 
              template. Only those areas are considered during classification. A block is rejected
              from the decision process if not all its pixels are masked. If no mask is passed, all
              blocks are considered valid.
    */
    class TemplateMatchCandidates {
    public:
        /** Set the source image. */
        void setSourceImage(const cv::Mat &image);
        
        /** Set the template size. */
        void setTemplateSize(cv::Size templateSize);

        /** Set the partition size. Specifies the number of blocks in x and y direction. */
        void setPartitionSize(cv::Size s);

        /** Initialize candidate search. */
        void initialize();

        /** 
            Find candidates.

            \param templ Template image.
            \param templMask Optional template mask.
            \param candidates Computed candidates mask.
            \param maxWeakErrors Max classification mismatches per channel.
            \param maxMeanDifference Max difference of patch / template mean before rejecting a candidate.
            \return Candidate mask.
        */
        void findCandidates(
            const cv::Mat &templ, 
            const cv::Mat &templMask, 
            cv::Mat &candidates,
            int maxWeakErrors = 3, 
            float maxMeanDifference = 20);

    private:

        /** Subdivides a size into a rectangle of blocks. */
        void computeBlockRects(
            cv::Size size, 
            cv::Size partitions, 
            std::vector< cv::Rect > &rects);

        /** Reject blocks depending on the template mask. */
        void removeInvalidBlocks(
            const cv::Mat &templMask, 
            std::vector< cv::Rect > &rects);

        /** Calculate the weak classifiers for the template, taking the mask into account. */
        void weakClassifiersForTemplate(
            const cv::Mat &templ, 
            const cv::Mat &templMask, 
            const std::vector< cv::Rect > &rects, 
            cv::Mat_<int> &classifiers, 
            cv::Scalar &mean);

        /** Compare the template classifiers to the classifiers generated from the given template position. */
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
        cv::Size _templateSize;
        cv::Size _partitionSize;
    };

    /**  
        Find candidate positions for template matching.

        This is a convinience method for using TemplateMatchCandidates.

        \param image Image to search in
        \param templ Template image
        \param templMask Optional template mask
        \param candidate A mask of possible candidates. If image size is W,H and template size is w,h
               the size of candidate will be W - w + 1, H - h + 1.
        \param partitionSize Number of blocks to subdivide template into
        \param maxWeakErrors Max classification mismatches per channel.
        \param maxMeanDifference Max difference of patch / template mean before rejecting a candidate.
    */
    void findTemplateMatchCandidates(
        cv::InputArray image,
        cv::InputArray templ,
        cv::InputArray templMask,
        cv::OutputArray candidates,
        cv::Size partitionSize = cv::Size(3,3),
        int maxWeakErrors = 3, 
        float maxMeanDifference = 20);

}
#endif