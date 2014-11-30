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
        Calculate the patch distance using cv::norm. 

      */
    class PatchDistanceCVNorm {
    public:
        PatchDistanceCVNorm()
            :_type(cv::NORM_L2SQR)
        {}

        PatchDistanceCVNorm(int type)
            :_type(type)
        {}

        inline float operator()(const cv::Mat &s, const cv::Mat &t, float previousDistance) const
        {
            return (float)cv::norm(s, t, _type);
        }
    private:
        int _type;
    };

    /**
        Compute dense approximate nearest neighbor fields.

        Implementation is based on 
        "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing", Barnes et al.
    */
    class PatchMatch {
    public:
        inline PatchMatch()
        {}

        void setSourceImage(const cv::Mat &img)
        {
            _input.source = img;
        }

        void setTargetImage(const cv::Mat &img)
        {
            _input.target = img;
        }

        void setTargetMask(const cv::Mat &mask)
        {
            _input.targetMask = mask;
        }

        void setPatchSize(int patchSize)
        {
            _input.patchSize = patchSize;
        }
        
        void setRandomSearchWindowSizeRatio(float alpha)
        {
            _input.alpha = alpha;
        }

        void initialize()
        {
            CV_Assert(
                (_input.source.type() == CV_MAKETYPE(CV_8U, 1) || _input.source.type() == CV_MAKETYPE(CV_8U, 3)) &&
                (_input.target.type() == _input.source.type()) &&
                (_input.target.size() == _input.source.size()) &&
                (_input.targetMask.empty() || (_input.targetMask.type() == CV_MAKETYPE(CV_8U, 1) && (_input.targetMask.size() == _input.target.size())))
            );

            _halfPatchSize = _input.patchSize / 2;

            _beginSource.x = _halfPatchSize;
            _beginSource.y = _halfPatchSize;
            _endSource.x = _input.source.cols  - _halfPatchSize - 1;
            _endSource.y = _input.source.rows  - _halfPatchSize - 1;

            _positions[0].create(_input.source.size());
            _positions[1].create(_input.source.size());
            _distances[0].create(_input.source.size());
            _distances[1].create(_input.source.size());

            _positions[0].setTo(cv::Vec2i(-1, -1));
            _positions[1].setTo(cv::Vec2i(-1, -1));
            _distances[0].setTo(std::numeric_limits<float>::max());
            _distances[1].setTo(std::numeric_limits<float>::max());

            if (_input.targetMask.empty()) {
                _targetMask.create(_input.target.size());
                _targetMask.setTo(255);                
            } else {
                _input.target.copyTo(_targetMask);
            }
            cv::rectangle(_targetMask, cv::Rect(0,0,_targetMask.cols, _targetMask.rows), cv::Scalar(0), _halfPatchSize * 2);

            // Initialize nearest neighbors through random assignment
            std::vector<cv::Vec2i> validTargetPositions;
            for (int y = 0; y < _targetMask.rows; ++y) {
                const uchar *row = _targetMask.ptr(y);
                for (int x = 0; x < _targetMask.cols; ++x) {
                    if (row[x])
                        validTargetPositions.push_back(cv::Vec2i(x, y));
                }
            }
            const int nValids = (int)validTargetPositions.size();
            cv::RNG rng(cv::getTickCount());

            for (int y = _beginSource.y; y < _endSource.y; ++y) {
                cv::Vec2i *positionRow = _positions[0].ptr<cv::Vec2i>(y);
                for (int x = _beginSource.x; x < _endSource.x; ++x) {
                    positionRow[x] = validTargetPositions[rng.uniform(0, nValids)];
                }
            }

            // Initialize propagation and search params

            _searchRadius = (float)std::max<int>(_input.target.cols, _input.target.rows);
            _offset.x = -1;
            _offset.y = -1;
        }

        cv::Mat_<cv::Vec2i> correspondences() const
        {
            return _positions[0];
        }

        cv::Mat_<float> distances() const
        {
            return _distances[0];
        }

        template<class DistanceMetric>
        void iterateOnce(const DistanceMetric &dm)
        {
            // Note when parallizing this method, take care of random numbers!
            cv::RNG rng(cv::getTickCount());

            TimerWithStats t;

            for (int y = _beginSource.y; y < _endSource.y; ++y) {
                
                const cv::Vec2i *inPositionRow = _positions[0].ptr<cv::Vec2i>(y);
                const float *inDistanceRow = _distances[0].ptr<float>(y);
                cv::Vec2i *outPositionRow = _positions[1].ptr<cv::Vec2i>(y);
                float *outDistanceRow = _distances[1].ptr<float>(y);

                for (int x = _beginSource.x; x < _endSource.x; ++x) {

                    float bestDist = inDistanceRow[x];
                    cv::Vec2i bestLocTarget = inPositionRow[x];
                    const cv::Mat sourcePatch = centeredPatch(_input.source, y, x, _halfPatchSize);

                    // Try to improve via propagation

                    const cv::Vec2i testLocSourceA(clamp(x + _offset.x, 0, _endSource.x - 1), y);
                    const cv::Vec2i testLocSourceB(x, clamp(y + _offset.y, 0, _endSource.y - 1));

                    const cv::Vec2i testLocTargetA = _positions[0].at<cv::Vec2i>(testLocSourceA[1], testLocSourceA[0]);
                    const cv::Vec2i testLocTargetB = _positions[0].at<cv::Vec2i>(testLocSourceB[1], testLocSourceB[0]);

                    if (testLocTargetA[0] != -1) {
                        float testD = dm(
                                        sourcePatch,
                                        centeredPatch(_input.target, testLocTargetA[1], testLocTargetA[0], _halfPatchSize),
                                        bestDist);
                        if (testD < bestDist) {
                            bestDist = testD;
                            bestLocTarget = testLocTargetA;
                        }
                    }

                    if (testLocTargetB[0] != -1) {
                        float testD = dm(
                                        sourcePatch,
                                        centeredPatch(_input.target, testLocTargetB[1], testLocTargetB[0], _halfPatchSize),
                                        bestDist);
                        if (testD < bestDist) {
                            bestDist = testD;
                            bestLocTarget = testLocTargetB;
                        }
                    }

                    t.measure(0);

                    // Try to improve via search

                    int multiplier = 0;
                    float searchRadius = _searchRadius * std::pow(_input.alpha, multiplier);
                    const cv::Vec2i searchStart = bestLocTarget;

                    do {
                        const cv::Vec2i testLocTarget(
                            clamp(searchStart[0] + (int)searchRadius * rng.uniform(-1, 1), 0, _targetMask.cols - 1),
                            clamp(searchStart[1] + (int)searchRadius * rng.uniform(-1, 1), 0, _targetMask.rows - 1)
                        );

                        if (_targetMask.at<uchar>(testLocTarget[1], testLocTarget[0])) {
                            float testD = dm(
                                        sourcePatch,
                                        centeredPatch(_input.target, testLocTarget[1], testLocTarget[0], _halfPatchSize),
                                        bestDist);

                            if (testD < bestDist) {
                                bestDist = testD;
                                bestLocTarget = testLocTarget;
                            }
                        }

                        ++multiplier;
                        searchRadius = _searchRadius * std::pow(_input.alpha, multiplier);
                    } while (searchRadius > 1.f);

                    t.measure(1);
                    
                    // Update with new best.

                    outPositionRow[x] = bestLocTarget;
                    outDistanceRow[x] = bestDist;
                    
                }
            }

            std::cout << t.total(0) << std::endl;
            std::cout << t.total(1) << std::endl;

            std::swap(_positions[0], _positions[1]);
            std::swap(_distances[0], _distances[1]);
            _offset.x *= -1;
            _offset.y *= -1;
        }

    private:
        struct UserSpecified {
            cv::Mat source;
            cv::Mat target;
            cv::Mat targetMask;
            int patchSize;
            float alpha;

            UserSpecified()
                : patchSize(9), alpha(0.5f)
            {}
        };

        template<class T>
        static inline T clamp(T x, T inclusiveMin, T inclusiveMax)
        {
            return std::max<T>(inclusiveMin, std::min<T>(x, inclusiveMax));
        }

        UserSpecified _input;
        cv::Mat_<cv::Vec2i> _positions[2];
        cv::Mat_<float> _distances[2];
        cv::Mat_<uchar> _targetMask;
        cv::Point _offset;
        cv::Point _beginSource, _endSource;
        int _halfPatchSize;
        float _searchRadius;
    };



    

}
#endif