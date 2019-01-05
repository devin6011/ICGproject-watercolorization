#ifndef PERLINNOISE_HPP
#define PERLINNOISE_HPP

#include <opencv2/opencv.hpp>
using namespace cv;


void getPerlinNoise(Mat& noise, const int row, const int col, const double scale = 0.05);


#endif
