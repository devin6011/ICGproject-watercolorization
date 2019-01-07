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

#include "ColorAdjustment.hpp"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <cfloat>

using namespace std;
using namespace cv;

void adjustColor(Mat &image, const int style) {

	//convert to CIE Lab
	image.convertTo(image, CV_32F, 1.0 / 255.0);
	cvtColor(image, image, COLOR_BGR2Lab);

	//calculate mean and standard deviation
	Scalar mean, std;
	meanStdDev(image, mean, std);

	// read pretrained model
	ifstream ifs("model");

	if(!ifs) {
		cerr << "Cannot read the style model" << endl;
		return;
	}

	string line;

	Mat centers;

	while(getline(ifs, line)) {

		istringstream iss(line);
		Mat center(1, 6, CV_32F);

		for(int i = 0; i < 6; ++i) {
			iss >> center.at<float>(i);
		}

		centers.push_back(center);
	}

	ifs.close();

	//select style
	Scalar targetMean, targetStd;

	if(style == -1) {
		//find best match style
		double bestDistance = DBL_MAX;
		int bestStyle;

		for(int i = 0; i < 20; ++i) {
			Mat candidate = centers.row(i);
			Mat source = (Mat_<float>(1, 6) << mean[0], mean[1], mean[2], std[0], std[1], std[2]);

			double distance = norm(candidate, source);

			if(distance < bestDistance) {
				bestStyle = i;
				bestDistance = distance;
			}
		}

		targetMean = Scalar(centers.at<float>(bestStyle, 0), centers.at<float>(bestStyle, 1), centers.at<float>(bestStyle, 2));
		targetStd = Scalar(centers.at<float>(bestStyle, 3), centers.at<float>(bestStyle, 4), centers.at<float>(bestStyle, 5));
	}
	else {
		targetMean = Scalar(centers.at<float>(style, 0), centers.at<float>(style, 1), centers.at<float>(style, 2));
		targetStd = Scalar(centers.at<float>(style, 3), centers.at<float>(style, 4), centers.at<float>(style, 5));
	}

	//color transfer
	image = image - mean;
	Mat3b image3b = image;
	image3b = image3b.mul(Scalar(targetStd[0]/std[0], targetStd[1]/std[1], targetStd[2]/std[2]));
	image = image + targetMean;
	
	//convert back to BGR
	cvtColor(image, image, COLOR_Lab2BGR);
	image.convertTo(image, CV_8UC3, 255.0);

}
