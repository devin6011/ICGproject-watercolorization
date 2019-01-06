#include "WetInWet.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace cv;

const float maxDistance = 3.0;

void applyWetInWet(Mat& image, const Mat& segment, const Mat& boundary, const Mat& gradientX, const Mat& gradientY) {
	RNG rng; // the magic power of Mr. Tang

	// Scatter some seeds around the boundaries on the brighter side
#pragma omp parallel for
	for(int i = 0; i < image.rows; ++i) {
		for(int j = 0; j < image.cols; ++j) {
			if(boundary.at<uchar>(i, j) == 1) {

				short dx = gradientX.at<short>(i, j), dy = gradientY.at<short>(i, j);
				int mag = sqrt(dx * dx + dy * dy);

				Point frontPoint(j + dx * 3 / mag, i + dy * 3 / mag);
				Point backPoint(j - dx * 3 / mag, i - dy * 3 / mag);
				if(frontPoint.x < 0 || frontPoint.x >= image.cols || frontPoint.y < 0 || frontPoint.y >= image.rows) {
					continue;
				}
				if(backPoint.x < 0 || backPoint.x >= image.cols || backPoint.y < 0 || backPoint.y >= image.rows) {
					continue;
				}

				Vec3b backColor = segment.at<Vec3b>(backPoint);

				float scatterDistance = rng.uniform(0.0f, maxDistance);

				//scatter on the brighter side
				Point destPoint(j + dx * scatterDistance / mag, i + dy * scatterDistance / mag);
				if(destPoint.x < 0 || destPoint.x >= image.cols || destPoint.y < 0 || destPoint.y >= image.rows) {
					continue;
				}
				image.at<Vec3b>(destPoint) = backColor;
			}
		}
	}

	//apply filter with an ellipse-shaped kernel along the normal vector
	Mat kernel = Mat::zeros(15, 15, CV_32F);

	for(int i = 0; i < 8; ++i) {
		kernel.at<float>(7, i) = (i + 1) / 8.0;
		kernel.at<float>(6, i) = (i + 1) / 16.0;
		kernel.at<float>(8, i) = (i + 1) / 16.0;
	}

	for(int i = 8; i < 15; ++i) {
		kernel.at<float>(7, i) = (15 - i) / 8.0;
		kernel.at<float>(6, i) = (15 - i) / 16.0; 
		kernel.at<float>(8, i) = (15 - i) / 16.0; 
	}

	Mat outputImage = image.clone();
	//imshow("scatter", image);

#pragma omp parallel for
	for(int i = 0; i < image.rows; ++i) {
		for(int j = 0; j < image.cols; ++j) {
			if(boundary.at<uchar>(i, j) == 1) {

				short dx = gradientX.at<short>(i, j), dy = gradientY.at<short>(i, j);
				float angle = fastAtan2(dy, dx);
				int mag = sqrt(dx * dx + dy * dy);

				for(int step = 0; step <= 15; ++step) {
					Point destPoint(j + dx * step / mag, i + dy * step / mag);
					if(destPoint.x < 0 || destPoint.x >= image.cols || destPoint.y < 0 || destPoint.y >= image.rows) {
						continue;
					}

					Mat rotatedKernel;
					Mat rotationMatrix = getRotationMatrix2D(Point(7, 7), angle, 1);
					warpAffine(kernel, rotatedKernel, rotationMatrix, kernel.size());

					Vec3f sum(0.0, 0.0, 0.0);
					float divisor = 0.0;

					for(int a = destPoint.y - 7; a <= destPoint.y + 7; ++a) {
						if(a < 0) {
							continue;
						}
						else if(a >= image.rows) {
							break;
						}

						for(int b = destPoint.x - 7; b <= destPoint.x + 7; ++b) {
							if(b < 0) {
								continue;
							}
							else if(b >= image.cols) {
								break;
							}

							sum += image.at<Vec3b>(a, b) * rotatedKernel.at<float>(a - destPoint.y + 7, b - destPoint.x + 7);
							divisor += rotatedKernel.at<float>(a - destPoint.y + 7, b - destPoint.x + 7);
						}
					}

					outputImage.at<Vec3b>(destPoint) = sum / divisor;
				}
			}
		}
	}

	image = outputImage;
}
