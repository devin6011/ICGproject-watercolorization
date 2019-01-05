#include "Granulation.hpp"
#include <opencv2/opencv.hpp>
#include "PerlinNoise.hpp"
#include <iostream>
using namespace std;
using namespace cv;

void applyGranulation(Mat& image) {
	Mat noise, noiseTexture = Mat::zeros(image.rows, image.cols, CV_32F);

	// generate noise texture
	getPerlinNoise(noise, image.rows, image.cols, 0.5);
	noiseTexture += noise * 4;
	getPerlinNoise(noise, image.rows, image.cols, 0.25);
	noiseTexture += noise * 2;
	getPerlinNoise(noise, image.rows, image.cols, 0.125);
	noiseTexture += noise;
	noiseTexture /= 7;
	//imshow("granulationNoise", noiseTexture);
	noiseTexture *= 0.25;
	noiseTexture += 0.875;

	Mat noiseTextureC3;
	Mat t[] = {noiseTexture, noiseTexture, noiseTexture};
	merge(t, 3, noiseTextureC3);

	image.convertTo(image, CV_32F, 1.0 / 255.0);

	image = image.mul(noiseTextureC3);

	image.convertTo(image, CV_8UC3, 255.0);
}
