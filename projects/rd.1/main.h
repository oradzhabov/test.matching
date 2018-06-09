#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <vector>

//#include <opencv2/core/core.hpp>
//#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/xfeatures2d.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <iterator>

namespace fs = std::experimental::filesystem;

namespace draw {
	void Lines(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb = CV_RGB(0, 0, 255), const int thickness = 1);
};

cv::Rect select_roi(const cv::Mat & img);


/**
* @param HoughAnglePrecissionDeg precission of lines detection. In radians
*/
std::vector<cv::Vec2f> shelfLines(const cv::Mat & img_scene, const double HoughAnglePrecission);

cv::Mat AdjustPerspective(const cv::Mat & img_scene, std::vector<cv::Vec2f> & lines, cv::Mat & perspectiveTranformMatrix);