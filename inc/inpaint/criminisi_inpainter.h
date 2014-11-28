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
      * Implementation of the exemplar based inpainting algorithm described in
      * "Object Removal by Exemplar-Based Inpainting", A. Criminisi et. al. 
      *
      * The implementation is not optimized for speed and trimmed towards educational 
      * purposes. Edge cases (i.e regions on the image border) are crudely handled by simply
      * discarding them.
      *
      * Besides, several robustness improvments are made by Christoph Heindl:
      *  - the template match error is calculated based on larger patch sizes than those
      *    used to infill. The reason behind this is to compare a larger portion of source
      *	   and target regions and thus to avoid visual artefacts.
      *
      *  - the search area for a patch to use for inpainting the target patch is first constrained
      *    to a local window around the target patch. If the best error within this search region is
      *	   too large, the search area is extended to the entire window. This is kind of a performance
      *    improvement with the reasoning that good matching exemplars are found nearby the target area.  
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
	    cv::Mat _image;
	    cv::Mat_<uchar> _targetRegion, _borderRegion, _sourceRegion;
	    cv::Mat_<float> _isophoteX, _isophoteY, _confidence, _borderGradX, _borderGradY;
	    int _halfPatchSize, _halfMatchSize;
	    int _startX, _startY, _endX, _endY;
    };

}
#endif