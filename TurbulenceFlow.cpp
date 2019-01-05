#include "TurbulenceFlow.hpp"
#include <opencv2/opencv.hpp>
#include "PerlinNoise.hpp"
#include <iostream>
using namespace std;
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
