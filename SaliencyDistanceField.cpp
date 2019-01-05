#include "SaliencyDistanceField.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <cfloat>
#include <algorithm>

using namespace std;
using namespace cv;

bool getBinarySaliencyMapDenseCut(const Mat& image, Mat& binarySaliencyMap) {
}

bool getBinarySaliencyMap(const Mat& image, Mat& binarySaliencyMap) {
	auto saliencyAlgorithm = saliency::StaticSaliencyFineGrained::create();
	//auto saliencyAlgorithm = saliency::StaticSaliencySpectralResidual::create();
	if(!saliencyAlgorithm) {
		cerr << "Error in the instantiation of the saliency algorithm" << endl;
		return 0;
	}

	Mat saliencyMap;
	if(saliencyAlgorithm->computeSaliency(image, saliencyMap)) {
		saliencyAlgorithm->computeBinaryMap(saliencyMap, binarySaliencyMap);

		//imshow("beforeS", saliencyMap);
		//imshow("beforeB", binarySaliencyMap);

		{
			Mat lookUpTable(1, 256, CV_8U, Scalar::all(0));
			lookUpTable.at<uchar>(0) = GC_PR_BGD;
			lookUpTable.at<uchar>(255) = GC_PR_FGD;
			LUT(binarySaliencyMap, lookUpTable, binarySaliencyMap);
		}

		double timer = (double)getTickCount();
		Mat bgdModel, fgdModel;
		grabCut(image, binarySaliencyMap, Rect(), bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
		grabCut(image, binarySaliencyMap, Rect(), bgdModel, fgdModel, 3, GC_EVAL);
		cout << "Grabcut : " << ((double)getTickCount() - timer) / getTickFrequency() << endl;

		{
			Mat lookUpTable(1, 256, CV_8U, Scalar::all(0));
			lookUpTable.at<uchar>(GC_BGD) = 0;
			lookUpTable.at<uchar>(GC_FGD) = 255;
			lookUpTable.at<uchar>(GC_PR_BGD) = 0;
			lookUpTable.at<uchar>(GC_PR_FGD) = 255;
			LUT(binarySaliencyMap, lookUpTable, binarySaliencyMap);
		}

		erode(binarySaliencyMap, binarySaliencyMap, Mat::ones(5, 5, CV_8U), Point(-1, -1), 1);
		dilate(binarySaliencyMap, binarySaliencyMap, Mat::ones(5, 5, CV_8U), Point(-1, -1), 2);

		//imshow("after", binarySaliencyMap);
	}
	else {
		cerr << "Error in the computation of the saliency" << endl;
		return 0;
	}

	//dilate(binarySaliencyMap, binarySaliencyMap, Mat::ones(5, 5, CV_8U), Point(-1, -1), 1);
	//imshow("saliencyM", saliencyMap);
	//imshow("saliencyB", binarySaliencyMap);

	return 1;
}

void getNormalizedDistanceField(const Mat& source, Mat& normalizedDistanceField) {
	//invert the binary image: black -> white, white-> black
	Mat invertedSource;
	bitwise_not(source, invertedSource);

	//calculate distance field and nearest zero pixel
	Mat distanceField, labels;
	distanceTransform(invertedSource, distanceField, labels, DIST_L2, DIST_MASK_3, DIST_LABEL_PIXEL);

	//normalize(distanceField, distanceField, 0, 1.0, NORM_MINMAX);
	//imshow("distance field", distanceField);

	double maximumIndex;
	minMaxIdx(labels, NULL, &maximumIndex);

	//get the coordinates of the nearest zero pixels
	vector<Point> nearestPointPosition((size_t)maximumIndex + 1);

	int nRows = labels.rows;
	int nCols = labels.cols;

	for(int i = 0; i < nRows; ++i) {
		int* ptrLabels = labels.ptr<int>(i);
		float* ptrDistanceField = distanceField.ptr<float>(i);

		for(int j = 0; j < nCols; ++j) {
			if(ptrDistanceField[j] == 0.0) {
				nearestPointPosition[ptrLabels[j]] = Point(j, i);
			}
		}
	}
	
	//calculate normalized distance field
	normalizedDistanceField = distanceField.clone();
	for(int i = 0; i < nRows; ++i) {
		int* ptrLabels = labels.ptr<int>(i);
		float* ptrDistanceField = distanceField.ptr<float>(i);
		float* ptrNormalizedDistanceField = normalizedDistanceField.ptr<float>(i);

		for(int j = 0; j < nCols; ++j) {
			if(ptrDistanceField[j] == 0.0) {
				continue;
			}

			Point nearestPoint = nearestPointPosition[ptrLabels[j]];

			Vec2f distance(j - nearestPoint.x, i - nearestPoint.y);

			float t = FLT_MAX; // parameter of ray

			if(distance[0] > 0.0f) { //right
				t = min(t, (float)(nCols - nearestPoint.x - 1) / distance[0]);
			}
			else if(distance[0] < 0.0f) { //left
				t = min(t, -(float)(nearestPoint.x) / distance[0]);
			}

			if(distance[1] > 0.0f) { //bottom
				t = min(t, (float)(nRows - nearestPoint.y - 1) / distance[1]);
			}
			else if(distance[1] < 0.0f) { //top
				t = min(t, -(float)(nearestPoint.y) / distance[1]);
			}

			Point2f borderPoint2f = Vec2f(nearestPoint.x, nearestPoint.y) + t * distance;
			Point borderPoint(borderPoint2f.x, borderPoint2f.y);

			ptrNormalizedDistanceField[j] = ptrDistanceField[j] / distanceField.at<float>(borderPoint.y, borderPoint.x);
			ptrNormalizedDistanceField[j] = min(1.0f, ptrNormalizedDistanceField[j]);
		}
	}
	//imshow("normalized distance field", normalizedDistanceField);
}

void getSaliencyDistanceField(const Mat& image, Mat& saliencyDistanceField) {
	Mat binarySaliencyMap;
	//getBinarySaliencyMapDenseCut(image, binarySaliencyMap);
	getBinarySaliencyMap(image, binarySaliencyMap);
	getNormalizedDistanceField(binarySaliencyMap, saliencyDistanceField);

	//smooth discontinuous artifacts
	for(int i = 0; i < 10; ++i) {
		GaussianBlur(saliencyDistanceField, saliencyDistanceField, Size(5, 5), 0, 0);
	}
}
