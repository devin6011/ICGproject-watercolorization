#ifndef SALIENCYDISTANCEFIELD_HPP
#define SALIENCYDISTANCEFIELD_HPP

#include <opencv2/opencv.hpp>
using namespace cv;

bool getBinarySaliencyMapDenseCut(const Mat& image, Mat& binarySaliencyMap);
bool getBinarySaliencyMap(const Mat& image, Mat& binarySaliencyMap);

void getNormalizedDistanceField(const Mat& source, Mat& normalizedDistanceField);

void getSaliencyDistanceField(const Mat& image, Mat& saliencyDistanceField);



#endif
