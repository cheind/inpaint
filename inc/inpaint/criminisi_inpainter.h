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

#ifndef INPAINT_CRIMINISI_INPAINTER_H
#define INPAINT_CRIMINISI_INPAINTER_H

#include <inpaint/template_match_candidates.h>
#include <opencv2/core/core.hpp>

namespace Inpaint {

    /**
        Implementation of the exemplar based inpainting algorithm described in
        "Object Removal by Exemplar-Based Inpainting", A. Criminisi et. al.

        Changes made by the author with respect to the original paper:
            - the template match error is calculated based on larger patch sizes than those
              used to infill. The reason behind this is to compare a larger portion of source
              and target regions and thus to avoid visual artefacts.

            - the search for the best matching spot of the patch position to be inpainted
              is accelerated by TemplateMatchCandidates.

        Please note edge cases (i.e regions on the image border) are crudely handled by simply
        discarding them.

      */
    class CriminisiInpainter {
    public:

        /** Empty constructor */
        CriminisiInpainter();

        /** Set the image to be inpainted. */
        void setSourceImage(const cv::Mat &bgrImage);

        /** Set the mask that describes the region inpainting can copy from. */
        void setSourceMask(const cv::Mat &mask);

        /** Set the mask that describes the region to be inpainted. */
        void setTargetMask(const cv::Mat &mask);

        /** Set the patch size. */
        void setPatchSize(int s);

        /** Initialize inpainting. */
        void initialize();

        /** True if there are more steps to perform. */
        bool hasMoreSteps();

        /** Perform a single step (i.e fill one patch) and return the updated information. */
        void step();

        /** Access the current state of the inpainted image. */
        cv::Mat image() const;

        /** Access the current state of the target region. */
        cv::Mat targetRegion() const;
    private:

        /** Updates the fill-front which is the border between filled and unfilled regions. */
        void updateFillFront();

        /** Find patch on fill front with highest priortiy. This will be the patch to be inpainted in this step. */
        cv::Point findTargetPatchLocation();

        /** For a given patch to inpaint, search for the best matching source patch to use for inpainting. */
        cv::Point findSourcePatchLocation(cv::Point targetPatchLocation, bool useCandidateFilter);

        /** Calculate the confidence for the given patch location. */
        float confidenceForPatchLocation(cv::Point p);

        /** Given that we know the source and target patch, propagate associated values from the source into the target region. */
        void propagatePatch(cv::Point target, cv::Point source);

        struct UserSpecified {
            cv::Mat image;
            cv::Mat sourceMask;
            cv::Mat targetMask;
            int patchSize;

            UserSpecified();
        };

        UserSpecified _input;

        TemplateMatchCandidates _tmc;
        cv::Mat _image, _candidates;
        cv::Mat_<uchar> _targetRegion, _borderRegion, _sourceRegion;
        cv::Mat_<float> _isophoteX, _isophoteY, _confidence, _borderGradX, _borderGradY;
        int _halfPatchSize, _halfMatchSize;
        int _startX, _startY, _endX, _endY;
    };

    /**
        Inpaint image.

        Implementation of the exemplar based inpainting algorithm described in
        "Object Removal by Exemplar-Based Inpainting", A. Criminisi et. al.
        
        \param image Image to be inpainted.
        \param targetMask Region to be inpainted.
        \param sourceMask Optional mask that specifies the region of the image to synthezise from. If left empty
               the entire image without the target mask is used.
        \param patchSize Patch size to use.
    */
    void inpaintCriminisi(
            cv::InputArray image,
            cv::InputArray targetMask,
            cv::InputArray sourceMask,
            int patchSize);

}
#endif
