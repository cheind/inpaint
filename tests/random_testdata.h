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

#ifndef INPAINT_RANDOM_TESTDATA_H
#define INPAINT_RANDOM_TESTDATA_H

#include <opencv2/opencv.hpp>

inline cv::Mat randomLinesImage(int imageSize, int nLines)
{
    cv::Mat m(imageSize, imageSize, CV_8UC1);
    m.setTo(0);

    cv::RNG rng(10);
    cv::Point pt1, pt2;
    for( int i = 0; i < nLines; ++i) {
        pt1.x = rng.uniform( 0, imageSize );
        pt1.y = rng.uniform( 0, imageSize );
        pt2.x = rng.uniform( 0, imageSize );
        pt2.y = rng.uniform( 0, imageSize );

        cv::line(m, pt1, pt2, rng.uniform(10, 255), rng.uniform(1, 10));
    }

    return m;
}

inline cv::Mat uniformRandomNoiseImage(int imageSize)
{
    cv::Mat m(imageSize, imageSize, CV_8UC1);
    cv::RNG rng(10);
    for( int i = 0; i < imageSize * imageSize; ++i) {
        m.at<uchar>(i) = rng.uniform(0,255);
    }

    return m;
}

inline cv::Mat randomBlock(const cv::Mat &image, cv::Rect &r)
{
    cv::RNG rng;

    int x1 = rng.uniform(0, image.cols - 1);
    int x2 = rng.uniform(0, image.cols - 1);
    
    if (x2 < x1) {
        std::swap(x1, x2);
    }

    int y1 = rng.uniform(0, image.rows - 1);
    int y2 = rng.uniform(0, image.rows - 1);
    
    if (y2 < y1) {
        std::swap(y1, y2);
    }

    r = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    return image(r);
}

inline cv::Mat shiftImage(const cv::Mat &img, int y, int x)
{
    cv::Mat out = cv::Mat::zeros(img.size(), img.type());
    img(cv::Rect(0,0,img.cols - x, img.rows - x)).copyTo(out(cv::Rect(x, y, img.cols - x, img.rows - x)));
    return out;
}

inline void randomGaussianBlobs(
        int blobs, // Not used if centers is provided.
        int samplesPerBlob,
        int dimensions,
        float clusterStdDev,
        cv::InputOutputArray centers_, cv::OutputArray features_, cv::OutputArray labels_,
        float minPosCenter = -20, float maxPosCenter = 20.f,
        uint64 randomSeed = cv::getTickCount())
{

    cv::Mat centers(blobs, dimensions, CV_64FC1);
    cv::Mat features(samplesPerBlob*blobs, dimensions, CV_64FC1);
    cv::Mat labels(1, samplesPerBlob*blobs, CV_32SC1);

    bool haveCenters = true;
    if (centers_.empty()) {
        haveCenters = false;
    } else {
        centers_.getMat().convertTo(centers, CV_64FC1);
    }

    cv::theRNG().state = randomSeed;

    int findex = 0;
    for (int i = 0; i < blobs; ++i) {
        if (!haveCenters) {
            cv::randu(centers.row(i), minPosCenter, maxPosCenter);
        }
        for (int f = 0; f < samplesPerBlob; ++f) {
            cv::randn(features.row(findex), 0, clusterStdDev);
            features.row(findex) += centers.row(i);
            labels.at<int>(0, findex) = i;
            ++findex;
        }
    }

    centers_.create(blobs, dimensions, CV_32FC1);
    centers.convertTo(centers_.getMat(), CV_32FC1);

    features_.create(samplesPerBlob*blobs, dimensions, CV_32FC1);
    features.convertTo(features_.getMat(), CV_32FC1);

    labels_.create(1, samplesPerBlob*blobs, CV_32SC1);
    labels.copyTo(labels_.getMat());

}


#endif
