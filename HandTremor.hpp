#ifndef HANDTREMOR_HPP
#define HANDTREMOR_HPP


#include <opencv2/opencv.hpp>
using namespace cv;

void applyHandTremor(Mat& image, const Mat& segment, const Mat& boundary);


#endif
