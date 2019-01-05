#ifndef WETINWET_HPP
#define WETINWET_HPP


#include <opencv2/opencv.hpp>
using namespace cv;

void applyWetInWet(Mat& image, const Mat& segment, const Mat& boundary, const Mat& gradientX, const Mat& gradientY);

#endif
