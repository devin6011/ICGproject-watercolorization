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

#include "Effects.hpp"
#include <opencv2/opencv.hpp>

#include "EdgeDarkening.hpp"
#include "Granulation.hpp"
#include "TurbulenceFlow.hpp"

using namespace cv;

void applyEffects(Mat& image) {
	applyEdgeDarkening(image);

	applyGranulation(image);

	applyTurbulenceFlow(image);
}
