#ifndef BOUNDARYCLASSIFICATION_HPP
#define BOUNDARYCLASSIFICATION_HPP

#include <opencv2/opencv.hpp>
using namespace cv;

void boundaryClassification(const Mat& image, const Mat& segment, const Mat& saliencyDistanceField, Mat& boundary, Mat& gradientX, Mat& gradientY);


#endif
