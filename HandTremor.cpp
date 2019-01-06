#include "HandTremor.hpp"
#include <opencv2/opencv.hpp>
#include "PerlinNoise.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void applyHandTremor(Mat& image, const Mat& segment, const Mat& boundary) {

	Mat noise1 = Mat::zeros(image.rows, image.cols, CV_32F);
	Mat noise2 = Mat::zeros(image.rows, image.cols, CV_32F);
	Mat noise3 = Mat::zeros(image.rows, image.cols, CV_32F);
	Mat noise4 = Mat::zeros(image.rows, image.cols, CV_32F);
	// generate noise texture
	{
		Mat noise;

		double base = 2.0;
		double divSum = 0;

		for(int i = 1; i <= 8; ++i) {
			getPerlinNoise(noise, image.rows * 2, image.cols * 2, base);
			base /= 2;
			noise1 += noise(Rect(0, 0, image.cols, image.rows)) / base;
			noise2 += noise(Rect(0, image.rows, image.cols, image.rows)) / base;
			noise3 += noise(Rect(image.cols, 0, image.cols, image.rows)) / base;
			noise4 += noise(Rect(image.cols, image.rows, image.cols, image.rows)) / base;
			divSum += 1.0 / base;
		}

		noise1 /= divSum;
		noise2 /= divSum;
		noise3 /= divSum;
		noise4 /= divSum;

		normalize(noise1, noise1, 0.25, 1.25, NORM_MINMAX);
		normalize(noise2, noise2, 0.25, 1.25, NORM_MINMAX);
		normalize(noise3, noise3, 0.25, 1.25, NORM_MINMAX);
		normalize(noise4, noise4, 0.25, 1.25, NORM_MINMAX);

		//imshow("handTremorNoise1", noise1);
		//imshow("handTremorNoise2", noise2);
		//imshow("handTremorNoise3", noise3);
		//imshow("handTremorNoise4", noise4);
	}

	//apply Hand Tremor Effect
	Mat outputImage = image.clone();
	//imshow("before", image);

	for(int i = 0; i < image.rows; ++i) {
		for(int j = 0; j < image.cols; ++j) {

			// no overlaps and gaps
			if(boundary.at<uchar>(i, j) == 2) {
				Point t1(j + noise1.at<float>(i, j), i + noise2.at<float>(i, j));
				if(t1.x < 0 || t1.x >= image.cols || t1.y < 0 || t1.y >= image.rows) {
					continue;
				}
				outputImage.at<Vec3b>(i, j) = image.at<Vec3b>(t1);
			}

			// with overlaps and gaps
			else if(boundary.at<uchar>(i, j) == 3) {
				Point t1(j + noise1.at<float>(i, j), i + noise2.at<float>(i, j));
				Point t2(j + noise3.at<float>(i, j), i + noise4.at<float>(i, j));

				Vec3b color1(255, 255, 255);
				Vec3b color2(255, 255, 255);

				if(!(t1.x < 0 || t1.x >= image.cols || t1.y < 0 || t1.y >= image.rows)) {
					if(segment.at<Vec3b>(i, j) == segment.at<Vec3b>(t1)) {
						color1 = image.at<Vec3b>(t1);
					}
				}
				if(!(t2.x < 0 || t2.x >= image.cols || t2.y < 0 || t2.y >= image.rows)) {
					if(segment.at<Vec3b>(i, j) == segment.at<Vec3b>(t2)) {
						color2 = image.at<Vec3b>(t2);
					}
				}

				outputImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255) - (Vec3b(255, 255, 255) - color1 + Vec3b(255, 255, 255) - color2);
			}
		}
	}

	image = outputImage;
}
