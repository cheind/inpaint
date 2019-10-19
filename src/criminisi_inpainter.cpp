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

#include <inpaint/criminisi_inpainter.h>
#include <inpaint/patch.h>
#include <inpaint/timer.h>
#include <inpaint/template_match_candidates.h>
#include <opencv2/opencv.hpp>

namespace Inpaint {

    const int PATCHFLAGS = PATCH_BOUNDS;

    CriminisiInpainter::UserSpecified::UserSpecified()
    {
        patchSize = 9;
    }

    CriminisiInpainter::CriminisiInpainter()
    {}

    void CriminisiInpainter::setSourceImage(const cv::Mat &bgrImage)
    {
        _input.image = bgrImage;
    }

    void CriminisiInpainter::setTargetMask(const cv::Mat &mask)
    {
        _input.targetMask = mask;
    }

    void CriminisiInpainter::setSourceMask(const cv::Mat &mask)
    {
        _input.sourceMask = mask;
    }

    void CriminisiInpainter::setPatchSize(int s)
    {
        _input.patchSize = s;
    }

    cv::Mat CriminisiInpainter::image() const
    {
        return _image;
    }

    cv::Mat CriminisiInpainter::targetRegion() const
    {
        return _targetRegion;
    }

    void CriminisiInpainter::initialize()
    {
        CV_Assert(
                    (_input.image.channels() == 3) &&
                    _input.image.depth() == CV_8U &&
                    _input.targetMask.size() == _input.image.size() &&
                    (_input.sourceMask.empty() || _input.targetMask.size() == _input.sourceMask.size()) &&
                    _input.patchSize > 0);

        _halfPatchSize = _input.patchSize / 2;
        _halfMatchSize = (int) (_halfPatchSize * 1.25f);

        _input.image.copyTo(_image);
        _input.targetMask.copyTo(_targetRegion);

        // Initialize regions
        cv::rectangle(_targetRegion, cv::Rect(0, 0, _targetRegion.cols, _targetRegion.rows), cv::Scalar(0), _halfMatchSize);

        _sourceRegion = 255 - _targetRegion;
        cv::rectangle(_sourceRegion, cv::Rect(0, 0, _sourceRegion.cols, _sourceRegion.rows), cv::Scalar(0), _halfMatchSize);
        cv::erode(_sourceRegion, _sourceRegion, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(_halfMatchSize*2+1, _halfMatchSize*2+1)));

        if (!_input.sourceMask.empty() && cv::countNonZero(_input.sourceMask) > 0) {
            _sourceRegion.setTo(0, (_input.sourceMask == 0));
        }

        // Initialize isophote values. Deviating from the original paper here. We've found that
        // blurring the image balances the data term and the confidence term better.
        cv::Mat blurred;
        cv::blur(_image, blurred, cv::Size(3,3));
        cv::Mat_<cv::Vec3f> gradX, gradY;
        cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

        _isophoteX.create(gradX.size());
        _isophoteY.create(gradY.size());

        for (int i = 0; i < gradX.rows * gradX.cols; ++i) {
            // Note the isophote corresponds to the gradient rotated by 90 degrees
            const cv::Vec3f &vx = gradX(i);
            const cv::Vec3f &vy = gradY(i);

            float x = (vx[0] + vx[1] + vx[2]) / (3 * 255);
            float y = (vy[0] + vy[1] + vy[2]) / (3 * 255);

            std::swap(x, y);
            x *= -1;

            _isophoteX(i) = x;
            _isophoteY(i) = y;
        }

        // Initialize confidence values
        _confidence.create(_image.size());
        _confidence.setTo(1);
        _confidence.setTo(0, _targetRegion);

        // Configure valid image region considered during algorithm
        _startX = _halfMatchSize;
        _startY = _halfMatchSize;
        _endX = _image.cols - _halfMatchSize - 1;
        _endY = _image.rows - _halfMatchSize - 1;

        // Setup template match performance improvement
        _tmc.setSourceImage(_image);
        _tmc.setTemplateSize(cv::Size(_halfMatchSize * 2 + 1, _halfMatchSize * 2 + 1));
        _tmc.setPartitionSize(cv::Size(3,3));
        _tmc.initialize();
    }

    bool CriminisiInpainter::hasMoreSteps()
    {
        return cv::countNonZero(_targetRegion) > 0;
    }

    void CriminisiInpainter::step()
    {
        // We also need an updated knowledge of gradients in the border region
        updateFillFront();

        // Next, we need to select the best target patch on the boundary to be inpainted.
        cv::Point targetPatchLocation = findTargetPatchLocation();

        // Determine the best matching source patch from which to inpaint.
        cv::Point sourcePatchLocation = findSourcePatchLocation(targetPatchLocation, true);
        if (sourcePatchLocation.x == -1)
            sourcePatchLocation = findSourcePatchLocation(targetPatchLocation, false);

        // Copy values
        propagatePatch(targetPatchLocation, sourcePatchLocation);
    }

    void CriminisiInpainter::updateFillFront()
    {
        // 2nd order derivative used to find border.
        cv::Laplacian(_targetRegion, _borderRegion, CV_8U, 3, 1, 0, cv::BORDER_REPLICATE);

        // Update confidence values along fill front.
        for (int y = _startY; y < _endY; ++y) {
            const uchar *bRow = _borderRegion.ptr(y);
            for (int x = _startX; x < _endX; ++x) {
                if (bRow[x] > 0) {
                    // Update confidence for border item
                    cv::Point p(x, y);
                    _confidence(p) = confidenceForPatchLocation(p);
                }
            }
        }
    }

    cv::Point CriminisiInpainter::findTargetPatchLocation()
    {
        // Sweep over all pixels in the border region and priorize them based on
        // a confidence term (i.e how many pixels are already known) and a data term that prefers
        // border pixels on strong edges running through them.

        float maxPriority = 0;
        cv::Point bestLocation(0, 0);

        _borderGradX.create(_targetRegion.size());
        _borderGradY.create(_targetRegion.size());
        cv::Sobel(_targetRegion, _borderGradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::Sobel(_targetRegion, _borderGradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

        for (int y = _startY; y < _endY; ++y) {
            const uchar *bRow = _borderRegion.ptr(y);
            const float *gxRow = _borderGradX.ptr<float>(y);
            const float *gyRow = _borderGradY.ptr<float>(y);
            const float *ixRow = _isophoteX.ptr<float>(y);
            const float *iyRow = _isophoteY.ptr<float>(y);
            const float *cRow = _confidence.ptr<float>(y);

            for (int x = _startX; x < _endX; ++x) {
                if (bRow[x] > 0) {

                    // Data term
                    cv::Vec2f grad(gxRow[x], gyRow[x]);
                    float dot = grad.dot(grad);

                    if (dot == 0) {
                        grad *= 0;
                    } else {
                        grad /= sqrtf(dot);
                    }

                    const float d = fabs(grad[0] * ixRow[x] + grad[1] * iyRow[x]) + 0.0001f;

                    // Confidence term
                    const float c = cRow[x];

                    // Priority of patch
                    const float prio = c * d;

                    if (prio > maxPriority) {
                        maxPriority = prio;
                        bestLocation = cv::Point(x,y);
                    }
                }
            }
        }

        return bestLocation;
    }

    float CriminisiInpainter::confidenceForPatchLocation(cv::Point p)
    {
        cv::Mat_<float> c = centeredPatch<PATCHFLAGS>(_confidence, p.y, p.x, _halfPatchSize);
        return (float)cv::sum(c)[0] / c.size().area();
    }

    cv::Point CriminisiInpainter::findSourcePatchLocation(cv::Point targetPatchLocation, bool useCandidateFilter)
    {
        cv::Point bestLocation(-1, -1);
        float bestError = std::numeric_limits<float>::max();

        cv::Mat_<cv::Vec3b> targetImagePatch = centeredPatch<PATCHFLAGS>(_image, targetPatchLocation.y, targetPatchLocation.x, _halfMatchSize);
        cv::Mat_<uchar> targetMask = centeredPatch<PATCHFLAGS>(_targetRegion, targetPatchLocation.y, targetPatchLocation.x, _halfMatchSize);

        cv::Mat invTargetMask = (targetMask == 0);
        if (useCandidateFilter)
            _tmc.findCandidates(targetImagePatch, invTargetMask, _candidates, 3, 10);
        
        int count = 0;
        for (int y = _startY; y < _endY; ++y) {
            for (int x = _startX; x < _endX; ++x) {

                // Note, candidates need to be corrected. Centered patch locations used here, top-left used with candidates.
                const bool shouldTest = (!useCandidateFilter || _candidates.at<uchar>(y - _halfMatchSize, x - _halfMatchSize)) &&
                        _sourceRegion.at<uchar>(y, x) > 0;

                if (shouldTest) {
                    ++count;
                    cv::Mat_<uchar> sourceMask = centeredPatch<PATCHFLAGS>(_sourceRegion, y, x, _halfMatchSize);
                    cv::Mat_<cv::Vec3b> sourceImagePatch = centeredPatch<PATCHFLAGS>(_image, y, x, _halfMatchSize);

                    float error = (float)cv::norm(targetImagePatch, sourceImagePatch, cv::NORM_L1, invTargetMask);

                    if (error < bestError) {
                        bestError = error;
                        bestLocation = cv::Point(x, y);
                    }
                }
            }
        }

        return bestLocation;
    }

    void CriminisiInpainter::propagatePatch(cv::Point target, cv::Point source)
    {
        cv::Mat_<uchar> copyMask = centeredPatch<PATCHFLAGS>(_targetRegion, target.y, target.x, _halfPatchSize);

        centeredPatch<PATCHFLAGS>(_image, source.y, source.x, _halfPatchSize).copyTo(
                    centeredPatch<PATCHFLAGS>(_image, target.y, target.x, _halfPatchSize),
                    copyMask);

        centeredPatch<PATCHFLAGS>(_isophoteX, source.y, source.x, _halfPatchSize).copyTo(
                    centeredPatch<PATCHFLAGS>(_isophoteX, target.y, target.x, _halfPatchSize),
                    copyMask);

        centeredPatch<PATCHFLAGS>(_isophoteY, source.y, source.x, _halfPatchSize).copyTo(
                    centeredPatch<PATCHFLAGS>(_isophoteY, target.y, target.x, _halfPatchSize),
                    copyMask);

        float cPatch = _confidence.at<float>(target);
        centeredPatch<PATCHFLAGS>(_confidence, target.y, target.x, _halfPatchSize).setTo(cPatch, copyMask);

        copyMask.setTo(0);
    }


    void inpaintCriminisi(
            cv::InputArray image,
            cv::InputArray targetMask,
            cv::InputArray sourceMask,
            int patchSize)
    {
        CriminisiInpainter ci;
        ci.setSourceImage(image.getMat());
        ci.setSourceMask(sourceMask.getMat());
        ci.setTargetMask(targetMask.getMat());
        ci.setPatchSize(patchSize);
        ci.initialize();
        
        while (ci.hasMoreSteps()) {
            ci.step();
        }

        ci.image().copyTo(image.getMat());
    }
}
