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

#include "EdgeDarkening.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

void applyEdgeDarkening(Mat& image) {
	Mat imageGray;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
	
	Mat edge;
	Mat gradientX, gradientY;
	Mat absGradientX, absGradientY;

	Scharr(imageGray, gradientX, CV_16S, 1, 0);
	Scharr(imageGray, gradientY, CV_16S, 0, 1);

	//converting back to CV_8U
	convertScaleAbs(gradientX, absGradientX);
	convertScaleAbs(gradientY, absGradientY);

	addWeighted(absGradientX, 0.5, absGradientY, 0.5, 0, edge);
	threshold(edge, edge, 80, 255, THRESH_BINARY | THRESH_OTSU);
	
	// Compute the edge color gradient
	Mat blurredEdge;
	blur(edge, blurredEdge, Size(5, 5));
	dilate(blurredEdge, blurredEdge, Mat::ones(5, 5, CV_8U), Point(-1, -1), 2);
	blur(blurredEdge, blurredEdge, Size(11, 11));
	addWeighted(edge, 0.1, blurredEdge, 0.7, 0, edge);
	GaussianBlur(edge, edge, Size(5, 5), 0, 0);

	edge.convertTo(edge, CV_32F, 1.0 / 255.0);
	edge *= 0.3;
	edge = 1.1 - edge;

	//imshow("edge detection", edge);

	Mat edgeC3;
	Mat t[] = {edge, edge, edge};
	merge(t, 3, edgeC3);

	image.convertTo(image, CV_32F, 1.0 / 255.0);

	image = image.mul(edgeC3);

	image.convertTo(image, CV_8UC3, 255.0);
}
