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

#include <inpaint/template_match_candidates.h>
#include <inpaint/patch.h>
#include <inpaint/integral.h>
#include <inpaint/timer.h>
#include <opencv2/opencv.hpp>

namespace Inpaint {

    void TemplateMatchCandidates::setSourceImage(const cv::Mat &image)
    {
        CV_Assert(
                    (image.channels() == 1 || image.channels() == 3) &&
                    (image.depth() == CV_8U)
                    );

        _image = image;
    }

    void TemplateMatchCandidates::setTemplateSize(cv::Size templateSize)
    {
        _templateSize = templateSize;
    }

    void TemplateMatchCandidates::setPartitionSize(cv::Size partitionSize)
    {
        _partitionSize = partitionSize;
    }

    void TemplateMatchCandidates::initialize()
    {
        std::vector< cv::Mat_<uchar> > imageChannels;
        cv::split(_image, imageChannels);
        const size_t nChannels = imageChannels.size();

        _integrals.resize(nChannels);
        for (size_t i = 0; i < nChannels; ++i) {
            cv::integral(imageChannels[i], _integrals[i]);
        }
        
        _blocks.clear();
        computeBlockRects(_templateSize, _partitionSize, _blocks);
    }


    void TemplateMatchCandidates::findCandidates(
            const cv::Mat &templ,
            const cv::Mat &templMask,
            cv::Mat &candidates,
            int maxWeakErrors,
            float maxMeanDifference)
    {
        CV_Assert(
                    templ.type() == CV_MAKETYPE(CV_8U, _integrals.size()) &&
                    templ.size() == _templateSize &&
                    (templMask.empty() || templMask.size() == _templateSize));

        candidates.create(
                    _image.size().height - templ.size().height + 1,
                    _image.size().width - templ.size().width + 1,
                    CV_8UC1);
        candidates.setTo(255);

        std::vector< cv::Rect > blocks = _blocks;
        removeInvalidBlocks(templMask, blocks);

        cv::Mat_<int> referenceClass;
        cv::Scalar templMean;
        weakClassifiersForTemplate(templ, templMask, blocks, referenceClass, templMean);
        
        // For each channel we loop over all possible template positions and compare with classifiers.
        for (size_t i = 0; i < _integrals.size(); ++i)
        {
            cv::Mat_<int> &integral = _integrals[i];
            const int *referenceClassRow = referenceClass.ptr<int>(static_cast<int>(i));

            // For all template positions ty, tx (top-left template position)
            for (int ty = 0; ty < candidates.rows; ++ty)
            {
                uchar *outputRow = candidates.ptr<uchar>(ty);

                for (int tx = 0; tx < candidates.cols; ++tx)
                {
                    if (!outputRow[tx])
                        continue;
                    
                    outputRow[tx] = compareWeakClassifiers(
                                integral,
                                tx, ty,
                                templ.size(),
                                blocks,
                                referenceClassRow,
                                (float)templMean[static_cast<int>(i)],
                            maxMeanDifference,
                            maxWeakErrors);
                }
            }
        }
    }

    void TemplateMatchCandidates::weakClassifiersForTemplate(
            const cv::Mat &templ,
            const cv::Mat &templMask,
            const std::vector< cv::Rect > &rects,
            cv::Mat_<int> &classifiers,
            cv::Scalar &mean)
    {
        const int nChannels = templ.channels();
        classifiers.create(nChannels, (int)rects.size());

        // Note we use cv::mean here to make use of mask.
        mean = cv::mean(templ, templMask);

        for (int x = 0; x < (int)rects.size(); ++x) {
            cv::Scalar blockMean = cv::mean(templ(rects[x]), templMask.empty() ? cv::noArray() : templMask(rects[x]));
            
            for (int y = 0; y < nChannels; ++y) {
                classifiers(y, x) = blockMean[y] > mean[y] ? 1 : -1;
            }
        }
    }

    uchar TemplateMatchCandidates::compareWeakClassifiers(
            const cv::Mat_<int> &i,
            int x, int y,
            cv::Size templSize,
            const std::vector< cv::Rect > &blocks,
            const int *compareTo,
            float templateMean,
            float maxMeanDiff, int maxWeakErrors)
    {
        const int *topRow = i.ptr<int>(y);
        const int *bottomRow = i.ptr<int>(y + templSize.height); // +1 required for integrals

        // Mean of image under given template position
        const float posMean = (bottomRow[x + templSize.width] - bottomRow[x] - topRow[x + templSize.width] + topRow[x]) / (1.f * templSize.area());

        if  (std::abs(posMean - templateMean) > maxMeanDiff)
            return 0;

        // Evaluate means of sub-blocks
        int sumErrors = 0;
        for (size_t r = 0; r < blocks.size(); ++r)
        {
            const cv::Rect &b = blocks[r];

            int ox = x + b.x;
            int oy = y + b.y;

            const int *topRow = i.ptr<int>(oy);
            const int *bottomRow = i.ptr<int>(oy + b.height);

            const float blockMean = (bottomRow[ox + b.width] - bottomRow[ox] - topRow[ox + b.width] + topRow[ox]) / (1.f * b.width * b.height);
            const int c = blockMean > posMean ? 1 : -1;
            sumErrors += (c != compareTo[r]) ? 1 : 0;

            if (sumErrors > maxWeakErrors)
                return 0;
        }

        return 255;
    }

    void TemplateMatchCandidates::computeBlockRects(cv::Size size, cv::Size partitions, std::vector< cv::Rect > &rects)
    {
        rects.clear();

        const int blockWidth = size.width / partitions.width;
        const int blockHeight = size.height / partitions.height;
        

        if (blockWidth == 0 || blockHeight == 0) {
            rects.push_back(cv::Rect(0, 0, size.width, size.height));
        } else {
            // Note: last row/column of blocks might be of different shape to fill up entire size.
            const int lastBlockWidth = size.width - blockWidth * (partitions.width - 1);
            const int lastBlockHeight = size.height - blockHeight * (partitions.height - 1);
            
            for (int y = 0; y < partitions.height; ++y) {
                bool lastY = (y == partitions.height - 1);
                for (int x = 0; x < partitions.width; ++x) {
                    bool lastX = (x == partitions.width - 1);

                    rects.push_back(cv::Rect(
                                        x * blockWidth,
                                        y * blockHeight,
                                        lastX ? lastBlockWidth : blockWidth,
                                        lastY ? lastBlockHeight : blockHeight));
                }
            }
        }
    }

    void TemplateMatchCandidates::removeInvalidBlocks(const cv::Mat &templMask, std::vector< cv::Rect > &rects)
    {
        if (!templMask.empty()) {
            rects.erase(std::remove_if(rects.begin(), rects.end(), [&templMask](const cv::Rect &r) -> bool {
                cv::Mat block = templMask(r);
                return cv::countNonZero(block) != block.size().area();
            }), rects.end());
        }
    }


    void findTemplateMatchCandidates(
            cv::InputArray image,
            cv::InputArray templ,
            cv::InputArray templMask,
            cv::OutputArray candidates,
            cv::Size partitionSize,
            int maxWeakErrors,
            float maxMeanDifference)
    {
        TemplateMatchCandidates tmc;
        tmc.setSourceImage(image.getMat());
        tmc.setPartitionSize(partitionSize);
        tmc.setTemplateSize(templ.size());
        tmc.initialize();

        candidates.create(
                    image.size().height - templ.size().height + 1,
                    image.size().width - templ.size().width + 1,
                    CV_8UC1);

        tmc.findCandidates(templ.getMat(), templMask.getMat(), candidates.getMatRef(), maxWeakErrors, maxMeanDifference);
    }

}
