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
#include <opencv2/opencv.hpp>

namespace Inpaint {

    CriminisiInpainter::CriminisiInpainter()
	    : _patchSize(9)
    {}

    void CriminisiInpainter::setImage(const cv::Mat &bgrImage)
    {
	    _image = bgrImage.clone();
    }

    void CriminisiInpainter::setMask(const cv::Mat &mask)
    {
	    _fillRegion = mask.clone();
    }

    void CriminisiInpainter::setPatchSize(int s)
    {
	    _patchSize = s;
    }

    cv::Mat CriminisiInpainter::image() const
    {
	    return _image.clone();
    }

    void CriminisiInpainter::initialize()
    {
	    // Remove elements from target region that are on borders.
	    cv::rectangle(_fillRegion, cv::Rect(0, 0, _fillRegion.cols, _fillRegion.rows), cv::Scalar(0), _patchSize);

	    // Initialize regions
	    _sourceRegion = 255 - _fillRegion; 
	    _initialSourceRegion = _sourceRegion.clone();

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
		    // Note the isophote corresponds to the gradient rotated by 90°
		    const cv::Vec3f &vx = gradX(i);
		    const cv::Vec3f &vy = gradY(i);
		
		    float x = (vx[0] + vx[1] + vx[2]) / (3 * 255);
		    float y = (vy[0] + vy[1] + vy[2]) / (3 * 255);

		    float t = x;
		    x = -y;
		    y = t;

		    _isophoteX(i) = x;
		    _isophoteY(i) = y;
	    }

	    // Initialize confidence values
	    _confidence.create(_image.size());
	    _confidence.setTo(1);
	    _confidence.setTo(0, _fillRegion);

	    // Configure valid image region
	    int phalf = _patchSize / 2;
	
	    _startX = phalf * 2;
	    _startY = phalf * 2;
	    _endX = _image.cols - phalf * 2 - 1;
	    _endY = _image.rows - phalf * 2 - 1;
    }

    bool CriminisiInpainter::hasMoreSteps()
    {
	    return cv::countNonZero(_fillRegion) > 0;
    }

    void CriminisiInpainter::step(cv::Mat &updatedImage, cv::Mat &updatedMask)
    {	
	    // We also need an updated knowledge of gradients in the border region
	    updateFillFront();
	
	    // Next, we need to select the best target patch on the boundary to be inpainted.
	    cv::Point targetPatchLocation = findTargetPatchLocation();

	    // Determine the best matching source patch from which to inpaint.
	    cv::Point sourcePatchLocation = findSourcePatchLocation(targetPatchLocation);

	    // Copy values
	    propagatePatch(targetPatchLocation, sourcePatchLocation);

	    // Prepare output
	    _image.copyTo(updatedImage);
	    _fillRegion.copyTo(updatedMask);
    }

    void CriminisiInpainter::updateFillFront()
    {
	    cv::Laplacian(_fillRegion, _borderRegion, CV_8U, 3, 1, 0, cv::BORDER_REPLICATE);

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

	    _borderGradX.create(_sourceRegion.size());
	    _borderGradY.create(_sourceRegion.size());
	    cv::Sobel(_sourceRegion, _borderGradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
	    cv::Sobel(_sourceRegion, _borderGradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

	    const int ph = _patchSize / 2;

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
	    cv::Mat_<float> c = patchFromLocation(_confidence, p);
	    return (float)cv::sum(c)[0] / c.size().area();
    }

    cv::Point CriminisiInpainter::findSourcePatchLocation(cv::Point targetPatchLocation)
    {	
	    // Try to find a good match in a local window first
	    int startY = std::max<int>(_startY, targetPatchLocation.y - (int)(0.25 * _image.rows));
	    int startX = std::max<int>(_startX, targetPatchLocation.x - (int)(0.25 * _image.cols));
	    int endY = std::min<int>(_endY, targetPatchLocation.y + (int)(0.25 * _image.rows));
	    int endX = std::min<int>(_endX, targetPatchLocation.x + (int)(0.25 * _image.cols));

	    cv::Point location;
	    float error = findSourcePatchLocationInWindow(targetPatchLocation, cv::Point(startX, startY), cv::Point(endX, endY), location);

	    if (error > 80.f) {
		    error = findSourcePatchLocationInWindow(targetPatchLocation, cv::Point(_startX, _startY), cv::Point(_endX, _endY), location);
	    }

	    return location;
    }

    float CriminisiInpainter::findSourcePatchLocationInWindow(cv::Point targetPatchLocation, cv::Point begin, cv::Point end, cv::Point &best)
    {
	    int matchSize = (int)(_patchSize * 1.5);
	    cv::Mat_<cv::Vec3b> targetImagePatch = patchFromLocation(_image, targetPatchLocation, matchSize);
	    cv::Mat_<uchar> targetMask = patchFromLocation(_sourceRegion, targetPatchLocation, matchSize);

	    float bestError = std::numeric_limits<float>::max();

	    for (int y = begin.y; y < end.y; ++y) {
		    for (int x = begin.x; x < end.x; ++x) {
			    cv::Point p(x, y);
			    cv::Mat_<uchar> sourceMask = patchFromLocation(_initialSourceRegion, p, matchSize);
			    cv::Mat_<cv::Vec3b> sourceImagePatch = patchFromLocation(_image, p, matchSize);

			    if (cv::countNonZero(sourceMask) != sourceMask.size().area())
				    continue;
			
			    float error = calculateTemplateMatchError(targetImagePatch, targetMask, sourceImagePatch);
			
			    if (error < bestError) {
				    bestError = error;
				    best = p;
			    }
		    }
	    }

	    return sqrtf(bestError);
    }

    float CriminisiInpainter::calculateTemplateMatchError(const cv::Mat &targetImage, const cv::Mat &targetMask, const cv::Mat &sourceImage)
    {
	    if (targetImage.size() != sourceImage.size()) {
		    return std::numeric_limits<float>::max();
	    }

	    int count = 0;
	    float sum = 0.f;

	    for (int y = 0; y < targetImage.rows; ++y) {
		    const cv::Vec3b *rTargetImage = targetImage.ptr<cv::Vec3b>(y);
		    const cv::Vec3b *rSourceImage = sourceImage.ptr<cv::Vec3b>(y);
		    const uchar *rTargetMask = targetMask.ptr<uchar>(y);
		    for (int x = 0; x < targetImage.cols; ++x) {

			    if (rTargetMask[x]) {
				    const cv::Vec3f t = rTargetImage[x];
				    const cv::Vec3f s = rSourceImage[x];
				    const cv::Vec3f v = t - s;
				    sum += v.dot(v);
				    count += 1;
			    }
		    }
	    }

	    return count > 0 ? sum / count : std::numeric_limits<float>::max();
    }

    void CriminisiInpainter::propagatePatch(cv::Point target, cv::Point source)
    {
	    cv::Mat_<uchar> copyMask = patchFromLocation(_fillRegion, target);

	    patchFromLocation(_image, source).copyTo(
		    patchFromLocation(_image, target), 
		    copyMask);

	    patchFromLocation(_isophoteX, source).copyTo(
		    patchFromLocation(_isophoteX, target), 
		    copyMask);		

	    patchFromLocation(_isophoteY, source).copyTo(
		    patchFromLocation(_isophoteY, target), 
		    copyMask);

	    float cPatch = _confidence.at<float>(target);
	    patchFromLocation(_confidence, target).setTo(cPatch, copyMask);
	
	    copyMask.setTo(0);
	    patchFromLocation(_sourceRegion, target).setTo(255);
    }

    cv::Mat CriminisiInpainter::patchFromLocation(const cv::Mat &src, cv::Point p)
    {
	    return patchFromLocation(src, p, _patchSize);	
    }

    cv::Mat CriminisiInpainter::patchFromLocation(const cv::Mat &src, cv::Point p, int patchSize)
    {
	    const int ph = patchSize / 2;
	    const int topx = std::max<int>(p.x - ph, 0);
	    const int topy = std::max<int>(p.y - ph, 0);
	    const int bottomx = std::min<int>(p.x + ph + 1, src.cols - ph - 1);
	    const int bottomy = std::min<int>(p.y + ph + 1, src.rows - ph - 1);

	    return src(cv::Rect(topx, topy, bottomx - topx, bottomy - topy));
    }

}