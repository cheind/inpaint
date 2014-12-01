/**
   This file is part of Inpaint.

   Copyright Christoph Heindl 2014

   Ioobar is free software: you can redistribute it and/or modify
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

#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat offsetImage(cv::Mat &image, cv::Scalar border, int xoffset, int yoffset)
{
    cv::Mat temp(image.rows+2*yoffset,image.cols+2*xoffset,image.type(),border);
    cv::Mat roi(temp(cv::Rect(xoffset,yoffset,image.cols,image.rows)));
    image.copyTo(roi);
    return temp;
}

/** Main entry point */
int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cerr << argv[0] << " image.png" << std::endl;
		return -1;
	}

	cv::Mat inputImage = cv::imread(argv[1]);

    cv::Mat target = offsetImage(inputImage, cv::Scalar(0), 20, 20);
    cv::imshow("target", target);
    cv::imshow("source", inputImage);
    cv::waitKey();

    cv::Mat corrs, distances, progress;
    progress.create(inputImage.size(), CV_8UC3);

    Inpaint::patchMatch(inputImage, target, cv::noArray(), corrs, distances, 5, 0);

    for (size_t i = 0; i < 100; ++i) {


        for (int y = 0; y < inputImage.rows; ++y) {
            cv::Vec2i *cRow = corrs.ptr<cv::Vec2i>(y);
            cv::Vec3b *iRow = progress.ptr<cv::Vec3b>(y);
            for (int x = 0; x < inputImage.cols; ++x) {
                iRow[x] = target.at<cv::Vec3b>(cRow[x][1], cRow[x][0]);
            }
        }
        cv::imshow("progress", progress);
        cv::waitKey();

        Inpaint::patchMatch(inputImage, target, cv::noArray(), corrs, distances, 5, 2);


        
    }

    cv::waitKey(0);

	return 0;
}




