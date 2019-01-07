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

#include "BoundaryClassification.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace std;
using namespace cv;

inline int angle(const int a, const int b) {
	return min(abs(a - b), 180 - abs(a - b));
}

void boundaryClassification(const Mat& image, const Mat& saliencyDistanceField, Mat& boundary, Mat& gradientX, Mat& gradientY) {
	Mat imageGray, imageHSV;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
	cvtColor(image, imageHSV, COLOR_BGR2HSV);

	Scharr(imageGray, gradientX, CV_16S, 1, 0);
	Scharr(imageGray, gradientY, CV_16S, 0, 1);

	//converting back to CV_8U
	Mat absGradientX, absGradientY;
	convertScaleAbs(gradientX, absGradientX);
	convertScaleAbs(gradientY, absGradientY);

	addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, boundary);
	threshold(boundary, boundary, 80, 255, THRESH_BINARY | THRESH_OTSU);

	//imshow("edge detection", boundary);
	
	for(int i = 0; i < image.rows; ++i) {
		for(int j = 0; j < image.cols; ++j) {

			// is boundary
			if(boundary.at<uchar>(i, j) == 255) {

				short dx = gradientX.at<short>(i, j), dy = gradientY.at<short>(i, j);
				int mag = sqrt(dx * dx + dy * dy);
				dx = dx * 3 / mag, dy = dy * 3 / mag;

				Point frontPoint(j + dx, i + dy);
				Point backPoint(j - dx, i - dy);
				if(frontPoint.x < 0 || frontPoint.x >= image.cols || frontPoint.y < 0 || frontPoint.y >= image.rows) {
					boundary.at<uchar>(i, j) = 0;
					continue;
				}
				if(backPoint.x < 0 || backPoint.x >= image.cols || backPoint.y < 0 || backPoint.y >= image.rows) {
					boundary.at<uchar>(i, j) = 0;
					continue;
				}

				// wet-in-wet
				if(((absGradientX.at<uchar>(i, j) >= 240 && absGradientY.at<uchar>(i, j) >= 240 && 0) ||
					(saliencyDistanceField.at<float>(i, j) < 0.3 && angle(imageHSV.at<Vec3b>(frontPoint)[0], imageHSV.at<Vec3b>(backPoint)[0]) < 10) ||
					(saliencyDistanceField.at<float>(i, j) >= 0.3 && angle(imageHSV.at<Vec3b>(frontPoint)[0], imageHSV.at<Vec3b>(backPoint)[0]) < 45))) {
					boundary.at<uchar>(i, j) = 1;
				}
				// hand tremor: similar hues, no overlaps and gaps
				else if(abs((int)imageHSV.at<Vec3b>(frontPoint)[0] - (int)imageHSV.at<Vec3b>(backPoint)[0]) < 45 ||
						min(imageHSV.at<Vec3b>(frontPoint)[1], imageHSV.at<Vec3b>(backPoint)[1]) < 45 ||
						min(imageHSV.at<Vec3b>(frontPoint)[2], imageHSV.at<Vec3b>(backPoint)[2]) < 45) {
					boundary.at<uchar>(i, j) = 2;
				}
				// hand tremor: other boundary, with overlaps and gaps
				else {
					boundary.at<uchar>(i, j) = 3;
				}
			}
			else {
				boundary.at<uchar>(i, j) = 0;
			}
		}
	}
}
