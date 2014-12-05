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

#include <inpaint/patch_match.h>
#include <inpaint/patch.h>
#include <opencv2/opencv.hpp>
#include <limits>

namespace Inpaint {

    template<bool HasTargetMaskSupport>
    class PatchMatchDistanceFunctor {
    public:
        PatchMatchDistanceFunctor(int normType)
            :_normType(normType)
        {}

        inline double operator() (
            const cv::Mat &source, 
            const cv::Mat &target, const cv::Mat &targetMask, 
            cv::Point sc, cv::Point tc, 
            int halfPatchSize) const
        {
            // If we are on a target boundary, exit.
            if (isCenteredPatchCrossingBoundary(tc, halfPatchSize, target))
                return std::numeric_limits<double>::max();

            // Determine comparable rect.
            std::pair<cv::Rect, cv::Rect> rects = comparablePatchRegions(source, target, sc, tc, halfPatchSize);
            if (rects.first.area() == 0)
                return std::numeric_limits<double>::max();

            cv::Mat pSource = topLeftPatch(source, rects.first);
            cv::Mat pTarget = topLeftPatch(target, rects.second);

            // If a target mask was specified, ensure we don't try to read from unmasked areas.
            if (HasTargetMaskSupport) {
                cv::Mat pTargetMask = topLeftPatch(targetMask, rects.second);            
                if (cv::countNonZero(pTargetMask) != rects.second.area()) {
                    return std::numeric_limits<double>::max();
                }
            }

            return cv::norm(pSource, pTarget, _normType);
        }
    private:
        int _normType;
    };

    template<class Distance>
    inline void patchMatchPropagateForward(
        cv::Mat source, cv::Mat target, cv::Mat targetMask,
        cv::Mat corrs, cv::Mat distances,
        int halfPatchSize,
        const Distance &distance)
    {
        // When forward we look at left and up neighbor

        const cv::Vec2i offsets = cv::Vec2i(-1, -1);
        for (int y = 1; y < source.rows; y++) {
            cv::Vec2i *corrsRow = corrs.ptr<cv::Vec2i>(y);
            const cv::Vec2i *corrsRowOther = corrs.ptr<cv::Vec2i>(y + offsets[1]);

            double *distancesRow = distances.ptr<double>(y);
            for (int x = 1; x < source.cols; x++) {
                
                // In the serial version we don't propagate self. So if we already have an optimum we keep it.
                if (distancesRow[x] == 0)
                    continue;

                cv::Point curPos(x, y);
                cv::Vec2i bestCorr = corrsRow[x];
                double bestDist = distancesRow[x];

                cv::Vec2i nCorrX = corrsRow[x + offsets[0]] + cv::Vec2i(-offsets[0], 0);
                double d = distance(
                    source, target, targetMask,
                    curPos, nCorrX,
                    halfPatchSize);

                if (d < bestDist) {
                    bestDist = d;
                    bestCorr = nCorrX;
                }

                cv::Vec2i nCorrY = corrsRowOther[x] + cv::Vec2i(0, -offsets[1]);
                d = distance(
                    source, target, targetMask,
                    curPos, nCorrY,
                    halfPatchSize);

                if (d < bestDist) {
                    bestDist = d;
                    bestCorr = nCorrY;
                }

                distancesRow[x] = bestDist;
                corrsRow[x] = bestCorr;
            }
        }
    }

    
    template<class Distance>
    inline void patchMatchPropagateBackward(
        cv::Mat source, cv::Mat target, cv::Mat targetMask,
        cv::Mat corrs, cv::Mat distances,
        int halfPatchSize,
        const Distance &distance)
    {
        // Backward we try to propagate from right and down.
        
        const cv::Vec2i offsets = cv::Vec2i(1, 1);
        for (int y = source.rows - 2; y >= 0; --y) {
            cv::Vec2i *corrsRow = corrs.ptr<cv::Vec2i>(y);
            const cv::Vec2i *corrsRowOther = corrs.ptr<cv::Vec2i>(y + offsets[1]);

            double *distancesRow = distances.ptr<double>(y);
            for (int x = source.cols - 2; x >= 0; --x) {
                
                // In the serial version we don't propagate self. So if we already have an optimum we keep it.
                if (distancesRow[x] == 0)
                    continue;

                cv::Point curPos(x, y);
                cv::Vec2i bestCorr = corrsRow[x];
                double bestDist = distancesRow[x];

                cv::Vec2i nCorrX = corrsRow[x + offsets[0]] + cv::Vec2i(-offsets[0], 0);
                double d = distance(
                    source, target, targetMask,
                    curPos, nCorrX,
                    halfPatchSize);

                if (d < bestDist) {
                    bestDist = d;
                    bestCorr = nCorrX;
                }

                cv::Vec2i nCorrY = corrsRowOther[x] + cv::Vec2i(0, -offsets[1]);
                d = distance(
                    source, target, targetMask,
                    curPos, nCorrY,
                    halfPatchSize);

                if (d < bestDist) {
                    bestDist = d;
                    bestCorr = nCorrY;
                }

                distancesRow[x] = bestDist;
                corrsRow[x] = bestCorr;
            }
        }
    }

    template<class Distance>
    inline void patchMatchExponentialSearch(
        cv::Mat source, cv::Mat target, cv::Mat targetMask,
        cv::Mat corrs, cv::Mat distances,
        int halfPatchSize,
        const Distance &distance,
        double alpha,
        int maxRadius)
    {
        cv::RNG rng(cv::getTickCount());

        for (int y = 0; y < source.rows; y++) {
            cv::Vec2i *corrsRow = corrs.ptr<cv::Vec2i>(y);
            double *distancesRow = distances.ptr<double>(y);

            for (int x = 0; x < source.cols; x++) {

                if (distancesRow[x] == 0)
                    continue;

                cv::Point curPos(x, y);
                cv::Vec2i bestCorr = corrsRow[x];
                double bestDist = distancesRow[x];

                int multiplier = 0;
                int radius = static_cast<int>(maxRadius * std::pow(alpha, multiplier));

                while (radius > 1) {
                    int minX = bestCorr[0] - radius;
                    int maxX = bestCorr[0] + radius + 1;
                    int minY = bestCorr[1] - radius;
                    int maxY = bestCorr[1] + radius + 1;

                    minX = clampLower(minX, 0);
                    minY = clampLower(minY, 0);
                    maxX = clampUpper(maxX, target.cols);
                    maxY = clampUpper(maxY, target.rows);

                    cv::Vec2i testPos(rng.uniform(minX, maxX), rng.uniform(minY, maxY));

                    double d = distance(
                        source, target, targetMask,
                        curPos, testPos,
                        halfPatchSize);

                    if (d < bestDist) {
                        bestDist = d;
                        bestCorr = testPos;
                    }

                    ++multiplier;
                    radius = static_cast<int>(maxRadius * std::pow(alpha, multiplier));
                }

                distancesRow[x] = bestDist;
                corrsRow[x] = bestCorr;
            }
        }
    }
   
    template<class Distance>
    inline void patchMatchOnce(
        cv::Mat source, cv::Mat target, cv::Mat targetMask,
        cv::Mat corrs, cv::Mat distances,
        int halfPatchSize,
        const Distance &d, bool forward,
        double alpha, int maxRadius)
    {
        if (forward) {
            patchMatchPropagateForward(source, target, targetMask, corrs, distances, halfPatchSize, d);
        } else {
            patchMatchPropagateBackward(source, target, targetMask, corrs, distances, halfPatchSize, d);
        }
        patchMatchExponentialSearch(source, target, targetMask, corrs, distances, halfPatchSize, d, alpha, maxRadius);
    }

    template<class Distance>
    void patchMatch(
        cv::InputArray &source_, 
        cv::InputArray &target_, cv::InputArray &targetMask_, 
        cv::OutputArray &corrs_, cv::OutputArray &distances_,
        int halfPatchSize,
        int iterations,
        const Distance &distance)
    {
        CV_Assert(
                (source_.type() == CV_MAKETYPE(CV_8U, 1) || source_.type() == CV_MAKETYPE(CV_8U, 3)) &&
                (target_.type() == source_.type()) &&
                (targetMask_.empty() || (targetMask_.type() == CV_MAKETYPE(CV_8U, 1) && (targetMask_.size() == target_.size()))) &&
                (halfPatchSize > 0) &&
                (iterations >= 0)
        );

        cv::Mat source = source_.getMat();
        cv::Mat target = target_.getMat();
        cv::Mat targetMask = targetMask_.getMat();
        cv::Mat corrs, distances;

        // If corrspondences are provided treat them as prior knowledge.
        bool neededRandomInitialization = false;
        if (!corrs_.empty()) {
            // Assume we have prior guess / knowledge about correspondences
            CV_Assert(corrs_.type() == CV_MAKETYPE(CV_32S, 2) &&
                      corrs_.size() == source.size());
            corrs = corrs_.getMat();
        } else {
            // Otherwise perform random initialization
            neededRandomInitialization = true;
            cv::RNG rng(cv::getTickCount());
            
            corrs_.create(source.size(), CV_32SC2);
            corrs = corrs_.getMat();
            
            // Note we do not take care of targetMask here. So even when provided,
            // elements from source might point to regions of target that are unmasked.
            // These wrong assignments will be dealt with when calculating distances.
            for (int y = 0; y < source.rows; ++y) {
                cv::Vec2i *corrsRow = corrs.ptr<cv::Vec2i>(y);
                for (int x = 0; x < source.cols; ++x) {
                    corrsRow[x][0] = rng.uniform(halfPatchSize, target.cols - halfPatchSize);
                    corrsRow[x][1] = rng.uniform(halfPatchSize, target.rows - halfPatchSize);
                }
            }
        }

        // If distances are provided treat them as prior knowlegde. This only makes sense when 
        // concurrently passing prior correspondences.
        if (!distances_.empty()) {
            CV_Assert(!neededRandomInitialization &&
                distances_.size() == source.size() &&
                distances_.type() == CV_MAKETYPE(CV_64F, 1));

            distances = distances_.getMat();
        } else {
            distances_.create(source.size(), CV_64FC1);
            distances = distances_.getMat();

            // Initialize distances made by random initialization.
            for (int y = 0; y < source.rows; ++y) {
                cv::Vec2i *corrsRow = corrs.ptr<cv::Vec2i>(y);
                double *distancesRow = distances.ptr<double>(y);
                for (int x = 0; x < source.cols; ++x) {
                    double d = distance(
                        source, target, targetMask, 
                        cv::Point(x, y), corrsRow[x], 
                        halfPatchSize);
                    distancesRow[x] = d;
                }
            }
        }

        double alpha = 0.5;
        int maxSearchRadius = maximum(target.cols, target.rows);

        bool forward = true;
        for (int i = 0; i  < iterations; ++i) {
            patchMatchOnce(source, target, targetMask, corrs, distances, halfPatchSize, distance, forward, alpha, maxSearchRadius);
        }
    }

   
    void patchMatch(
        cv::InputArray &source_, 
        cv::InputArray &target_, cv::InputArray &targetMask_, 
        cv::InputOutputArray &corrs_, cv::InputOutputArray &distances_,
        int halfPatchSize,
        int iterations,
        int normType)
    {

        if (targetMask_.empty()) {
            patchMatch(source_, target_, targetMask_, corrs_, distances_, halfPatchSize, iterations, PatchMatchDistanceFunctor<false>(normType));
        } else {
            patchMatch(source_, target_, targetMask_, corrs_, distances_, halfPatchSize, iterations, PatchMatchDistanceFunctor<true>(normType));
        }
    }

}