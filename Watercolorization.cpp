#include <iostream>
#include <opencv2/opencv.hpp>

#include "ColorAdjustment.hpp"
#include "SaliencyDistanceField.hpp"
#include "Abstraction.hpp"
#include "BoundaryClassification.hpp"
#include "Effects.hpp"
#include "WetInWet.hpp"
#include "HandTremor.hpp"

using namespace std;
using namespace cv;

void printUsage() {
	cout << "Usage: watercolorization file [outputfilename] [flags]" << endl;
}

void printElapsedTime(double& timer, const string& stage) {
	cout << stage + " : " << (getTickCount() - timer) / getTickFrequency() << endl;
	timer = getTickCount();
}

void watercolorize(Mat& image) {
	double timer = getTickCount();
	double totalTimer = timer;

	adjustColor(image);
	printElapsedTime(timer, "AdjustColor");

	Mat saliencyDistanceField;
	getSaliencyDistanceField(image, saliencyDistanceField);
	printElapsedTime(timer, "SaliencyDistanceField");

	//imshow("saliecy", saliencyDistanceField);
	Mat segment;
	segmentation(image, segment);
	//image = segment.clone();
	printElapsedTime(timer, "Segmentation");
	//imshow("segmentation", segment);
	
	abstraction(image, segment, saliencyDistanceField);
	printElapsedTime(timer, "Abstraction");
	//imshow("abstraction", image);
	
	Mat boundary;
	Mat gradientX, gradientY;

	boundaryClassification(image, segment, saliencyDistanceField, boundary, gradientX, gradientY);
	printElapsedTime(timer, "BoundaryClassification");
	//imshow("boundary", boundary);

	{
		//Mat showBoundary;
		//Mat lookUpTable = Mat::zeros(1, 256, CV_8U);
		//lookUpTable.at<uchar>(1) = 255;
		//lookUpTable.at<uchar>(2) = 120;
		//lookUpTable.at<uchar>(3) = 50;
		//LUT(boundary, lookUpTable, showBoundary);
		//imshow("boundary", showBoundary);
	}

	applyHandTremor(image, segment, boundary);
	printElapsedTime(timer, "HandTremor");
	
	applyWetInWet(image, segment, boundary, gradientX, gradientY);
	printElapsedTime(timer, "WetInWet");

	applyEffects(image);
	printElapsedTime(timer, "Other Effects");

	//smooth the final result
	GaussianBlur(image, image, Size(3, 3), 0, 0);

	printElapsedTime(totalTimer, "Total");

	imshow("output", image);

	waitKey(0);
}

int main(int argc, char* argv[]) {

	if(argc < 2 || argc > 4) {
		printUsage();
		return 0;
	}

	Mat image = imread(argv[1], IMREAD_COLOR);

	if(!image.data) {
		cerr << "Cannot load image." << endl;
		return -1;
	}

	watercolorize(image);

	bool writeSuccess = false;
	try {
		if(argc == 2) {
			writeSuccess = imwrite("./output.jpg", image);
		}
		else {
			writeSuccess = imwrite(argv[2], image);
		}
	}
	catch (const cv::Exception& ex) {
		//so what?
		cerr << ex.what() << endl;
	}

	if(!writeSuccess) {
		cerr << "Cannot save image." << endl;
		return -1;
	}

	cout << "Done" << endl;

	return 0;
}
