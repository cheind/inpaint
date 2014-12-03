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

#include <inpaint/pyramid.h>
#include <opencv2/opencv.hpp>

namespace Inpaint {

    void imagePyramid(cv::InputArray image_, cv::OutputArrayOfArrays pyr_, cv::Size minimumSize, int interpolationType)
    {
        CV_Assert(pyr_.kind() == cv::_InputArray::STD_VECTOR_MAT );

        std::vector<cv::Mat>& images = *( std::vector<cv::Mat>*)pyr_.getObj();

        images.clear();        
        images.push_back(image_.getMat().clone());
        while (images.back().cols >= minimumSize.width * 2 && images.back().rows >= minimumSize.height * 2)
        {
            cv::Mat img;
            cv::resize(images.back(), img, cv::Size(), 0.5, 0.5, interpolationType);
            images.push_back(img);
        }
    }
}