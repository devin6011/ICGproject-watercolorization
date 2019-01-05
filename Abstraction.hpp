#ifndef ABSTRACTION_HPP
#define ABSTRACTION_HPP

#include <opencv2/opencv.hpp>
using namespace cv;

bool segmentation(const Mat& image, Mat& segment);
void abstraction(Mat& image, const Mat& segment, const Mat& saliencyDistanceField);



#endif
