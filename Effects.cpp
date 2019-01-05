#include "Effects.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#include "EdgeDarkening.hpp"
#include "Granulation.hpp"
#include "TurbulenceFlow.hpp"

using namespace std;
using namespace cv;

void applyEffects(Mat& image, const Mat& segment) {
	applyEdgeDarkening(image, segment);

	applyGranulation(image);

	applyTurbulenceFlow(image);
}
