#include "Effects.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#include "EdgeDarkening.hpp"
#include "Granulation.hpp"
#include "TurbulenceFlow.hpp"

using namespace std;
using namespace cv;

void applyEffects(Mat& image) {
	applyEdgeDarkening(image);

	applyGranulation(image);

	applyTurbulenceFlow(image);
}
