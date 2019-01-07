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

#include "PerlinNoise.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

// the following code segment is from https://github.com/yoggy/cv_perlin_noise
/*
 * Copyright (c) 2015 yoggy
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
class PerlinNoise {
public:
	int p[512];

	PerlinNoise() {
		int permutation[] = { 151, 160, 137, 91, 90, 15, 131, 13, 201,
			95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37,
			240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62,
			94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56,
			87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139,
			48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133,
			230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
			63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200,
			196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3,
			64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255,
			82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
			223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153,
			101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79,
			113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242,
			193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
			14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
			176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222,
			114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180 };

		for (int i = 0; i < 256; i++) {
			p[256 + i] = p[i] = permutation[i];
		}
	}

	double noise(const double &src_x, const double &src_y, const double &src_z) {
		int X = (int)floor(src_x) & 255;
		int Y = (int)floor(src_y) & 255;
		int Z = (int)floor(src_z) & 255;

		double x = src_x - floor(src_x);
		double y = src_y - floor(src_y);
		double z = src_z - floor(src_z);

		double u = fade(x);
		double v = fade(y);
		double w = fade(z);

		int A = p[X] + Y;
		int AA = p[A] + Z;
		int AB = p[A + 1] + Z;
		int B = p[X + 1] + Y;
		int BA = p[B] + Z;
		int BB = p[B + 1] + Z;

		double r = lerp(
			w,
			lerp(v,
				lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
				lerp(u, grad(p[AB], x, y - 1, z),
				grad(p[BB], x - 1, y - 1, z))),
			lerp(v,
				lerp(u, grad(p[AA + 1], x, y, z - 1),
				grad(p[BA + 1], x - 1, y, z - 1)),
				lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
				grad(p[BB + 1], x - 1, y - 1, z - 1))));

		return r;
	}

protected:
	double fade(double t) {
		return t * t * t * (t * (t * 6 - 15) + 10);
	}

	double lerp(double t, double a, double b) {
		return a + t * (b - a);
	}

	double grad(int hash, double x, double y, double z) {
		int h = hash & 15;
		double u = h < 8 ? x : y;
		double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
	}
};

PerlinNoise perlin_noise;;

cv::Mat CreatePerlinNoiseImage(const cv::Size &size, const double &scale)
{
	cv::Mat img;
	img.create(size, CV_8UC1);

	for (int y = 0; y < size.height; ++y) {
		for (int x = 0; x < size.width; ++x) {
			double p = perlin_noise.noise(x  * scale, y * scale, 0.0); // -1.0`1.0
			p = (p + 1.0) / 2.0; // 0.0`1.0			
			img.at<uchar>(cv::Point(x, y)) = (uchar)(p * 255);
		}
	}

	return img;
}
// The preceding code segment is from https://github.com/yoggy/cv_perlin_noise


void getPerlinNoise(Mat& noise, const int row, const int col, const double scale) {
	noise.create(row, col, CV_32F);
	noise.forEach<float>([&](float &p, const int* position) -> void {
		p = perlin_noise.noise(position[1] * scale, position[0] * scale, 0.0);
		p = (p + 1.0) / 2.0;
	});
}
