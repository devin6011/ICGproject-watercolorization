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

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

const int dataSize = 400;
const int clusterNumber = 20;

void shuffleRows(Mat& mat) {
	Mat output;

	Mat indices(mat.rows, 1, CV_32S);
	for(int i = 0; i < mat.rows; ++i)
		indices.at<int>(i) = i;

	randShuffle(indices);

	for(int i = 0; i < mat.rows; ++i)
		output.push_back(mat.row(indices.at<int>(i)));

	mat = output;
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		cout << "Usage: Clustering [Image Dir]" << endl;
		return 0;
	}

	Mat trainData;

	filesystem::path rootDirPath(argv[1]);
	auto imageDirPath = rootDirPath / "Original";

	vector<string> filepaths;

	for(auto &&p : filesystem::directory_iterator(imageDirPath)) {
		Mat image = imread(p.path().string(), IMREAD_COLOR);
		filepaths.push_back(p.path().string());

		image.convertTo(image, CV_32F, 1.0 / 255.0);
		cvtColor(image, image, COLOR_BGR2Lab);

		Mat mean, std;
		meanStdDev(image, mean, std);

		Mat featureVector;
		vconcat(mean, std, featureVector);
		
		trainData.push_back(featureVector.t());
	}

	cout << "Training Data Done." << endl;

	trainData.convertTo(trainData, CV_32F);
	shuffleRows(trainData);

	Mat labels, centers;
	double compactness = kmeans(trainData, clusterNumber, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1e-4), 1126, KMEANS_PP_CENTERS, centers);
	cout << "Kmeans Done.\nCompactness : " << compactness << endl;

	for(int i = 0; i < clusterNumber; ++i) {
		auto classDirPath = rootDirPath / to_string(i);
		filesystem::create_directory(classDirPath);
	}

	vector<int> symCnt(clusterNumber);

	for(int i = 0; i < dataSize; ++i) {
		int clusterLabel = labels.at<int>(i);
		symCnt[clusterLabel]++;

		auto symPath = rootDirPath / to_string(clusterLabel) / (to_string(symCnt[clusterLabel]) + ".jpg");

		if(filesystem::exists(symPath))
			filesystem::remove(symPath);

		filesystem::create_symlink(filepaths[i], symPath);
	}

	ofstream ofs(rootDirPath / "model");
	ofs << fixed;
	ofs.precision(9);

	for(int i = 0; i < clusterNumber; ++i) {
		for(int j = 0; j < 6; ++j) {
			if(j > 0)
				ofs << ' ';
			ofs << centers.at<float>(i, j);
		}
		ofs << '\n';
	}

	ofs.close();
	return 0;
}
