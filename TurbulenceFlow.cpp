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

#include "TurbulenceFlow.hpp"
#include <opencv2/opencv.hpp>
#include "PerlinNoise.hpp"
using namespace cv;

void applyTurbulenceFlow(Mat& image) {
	Mat noise, noiseTexture = Mat::zeros(image.rows, image.cols, CV_32F);

	// generate noise texture
	{
		double base = 1.5;
		double divSum = 0;

		for(int i = 1; i <= 8; ++i) {
			getPerlinNoise(noise, image.rows, image.cols, base);
			base /= 2;
			noiseTexture += noise / base;
			divSum += 1.0 / base;
		}

		noiseTexture /= divSum;
		//imshow("turbulenceNoise", noiseTexture);
	}
	noiseTexture += 0.65;

	Mat noiseTextureC3;
	Mat t[] = {noiseTexture, noiseTexture, noiseTexture};
	merge(t, 3, noiseTextureC3);

	image.convertTo(image, CV_32F, 1.0 / 255.0);

	image = image.mul(noiseTextureC3);

	image.convertTo(image, CV_8UC3, 255.0);
}
