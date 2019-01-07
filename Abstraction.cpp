/*
 * Watecolorization : An Implementation of The Paper,
 * "Towards Photo Watercolorization with Artistic Verisimilitude".
 * Copyright (C) 2019 devin6011
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "Abstraction.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/hfs.hpp>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace cv;

bool segmentation(const Mat& image, Mat& segment) {
	auto segmentationAlgorithm = hfs::HfsSegment::create(image.rows, image.cols, 0.02f, 30, 0.08f, 60, 0.3f, 32, 16);
	if(!segmentationAlgorithm) {
		cerr << "Error in the instantiation of the segmentation algorithm" << endl;
		return 0;
	}

	//To show the image, change the second argument to be true
	segment = segmentationAlgorithm->performSegmentCpu(image, true);

	//imshow("segment", segment);
	return 1;
}

/*
inline bool insideImage(const Mat& image, int i, int j) {
	return !(i < 0 || i >= image.rows || j < 0 || j >= image.cols);
}
*/

inline int clamp(int n, int a, int b) {
	if(n < a) return a;
	if(n > b) return b;
	return n;
}

void abstraction(Mat& image, const Mat& segment, const Mat& saliencyDistanceField) {
	Mat outputImage(image.size(), CV_8UC3);

#pragma omp parallel for
	for(int i = 0; i < image.rows; ++i) {
		for(int j = 0; j < image.cols; ++j) {
			int count = 0;
			Vec3f sum(0.0, 0.0, 0.0);

			int halfKernelSize = 2;
			bool nonSaliency = 0;

			float d = saliencyDistanceField.at<float>(i, j);
			if(d > 1e-8) {
				nonSaliency = 1;
				const int kernelSize = clamp(10 * (d + 0.3f), 4, 9);
				halfKernelSize = kernelSize / 2;
			}

			for(int a = i - halfKernelSize; a <= i + halfKernelSize; ++a) {
				if(a < 0) {
					continue;
				}
				else if(a >= image.rows) {
					break;
				}

				for(int b = j - halfKernelSize; b <= j + halfKernelSize; ++b) {
					if(b < 0) {
						continue;
					}
					else if(b >= image.cols) {
						break;
					}

					if(/*insideImage(image, a, b) &&*/ segment.at<Vec3b>(i, j) == segment.at<Vec3b>(a, b)) {
						sum += image.at<Vec3b>(a, b);
						++count;
					}
					else if(/*insideImage(image, a, b) &&*/ nonSaliency && fabs(saliencyDistanceField.at<float>(i, j) - saliencyDistanceField.at<float>(a, b)) < 0.3f * d) {
						sum += image.at<Vec3b>(a, b);
						++count;
					}
					else {
						sum += image.at<Vec3b>(i, j);
						++count;
					}
				}
			}

			outputImage.at<Vec3b>(i, j) = sum / count;
		}
	}

	image = outputImage;
}
