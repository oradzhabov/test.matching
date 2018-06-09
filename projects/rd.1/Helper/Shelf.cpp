#include "main.h"


void mergeRelatedLines(std::vector<cv::Vec2f> & lines, const cv::Mat &img, const double minDistance) {

	std::vector<cv::Vec2f>::iterator current;
	for (current = lines.begin(); current != lines.end(); current++) {

		if ((*current)[0] == 0 && (*current)[1] == -100) continue;

		const float p1 = (*current)[0];
		const float theta1 = (*current)[1];

		cv::Point pt1current, pt2current;
		if (theta1>CV_PI * 45 / 180 && theta1<CV_PI * 135 / 180) {
			pt1current.x = 0;
			pt1current.y = static_cast<int>(p1 / sin(theta1));

			pt2current.x = img.size().width;
			pt2current.y = static_cast<int>(-pt2current.x / tan(theta1) + p1 / sin(theta1));
		} else {
			pt1current.y = 0;
			pt1current.x = static_cast<int>(p1 / cos(theta1));

			pt2current.y = img.size().height;
			pt2current.x = static_cast<int>(-pt2current.y / tan(theta1) + p1 / cos(theta1));
		}
		//
		std::vector<cv::Vec2f>::iterator    pos;
		for (pos = lines.begin(); pos != lines.end(); pos++) {

			if (*current == *pos) continue;
			if (fabs((*pos)[0] - (*current)[0]) < minDistance && fabs((*pos)[1] - (*current)[1]) < CV_PI * 10 / 180) {
				const float p = (*pos)[0];
				const float theta = (*pos)[1];
				//
				cv::Point pt1, pt2;
				if ((*pos)[1]>CV_PI * 45 / 180 && (*pos)[1]<CV_PI * 135 / 180) {
					pt1.x = 0;
					pt1.y = static_cast<int>(p / sin(theta));
					pt2.x = img.size().width;
					pt2.y = static_cast<int>(-pt2.x / tan(theta) + p / sin(theta));
				} else {
					pt1.y = 0;
					pt1.x = static_cast<int>(p / cos(theta));
					pt2.y = img.size().height;
					pt2.x = static_cast<int>(-pt2.y / tan(theta) + p / cos(theta));
				}
				if (((double)(pt1.x - pt1current.x)*(pt1.x - pt1current.x) + (pt1.y - pt1current.y)*(pt1.y - pt1current.y)<64 * 64) &&
					((double)(pt2.x - pt2current.x)*(pt2.x - pt2current.x) + (pt2.y - pt2current.y)*(pt2.y - pt2current.y)<64 * 64)) {
					// Merge the two
					(*current)[0] = ((*current)[0] + (*pos)[0]) / 2;
					(*current)[1] = ((*current)[1] + (*pos)[1]) / 2;

					(*pos)[0] = 0;
					(*pos)[1] = -100;
				}
			}
		}
	}
	// Drop fault lines
	std::vector<cv::Vec2f> res;
	std::copy_if(lines.begin(), lines.end(), std::back_inserter(res), [](cv::Vec2f i) {return i[1] > -100;});

	// Sort by first argument(collect lines from top to bottom)
	std::sort(res.begin(), res.end(), [](const cv::Vec2f & a, const cv::Vec2f & b) -> bool {return a[0] < b[0];});

	lines = res;
}

double findPeriod(const std::vector<double> & arg) {
	if (arg.size() < 2) return 0;

	cv::Mat matDist(1, arg.size() - 1, CV_64FC1);
	double * dist = matDist.ptr<double>(0);
	for (int i = 0; i + 1 < arg.size(); i++) {
		double d = arg[i + 1] - arg[i];
		dist[i] = d;

		//printf("%f\n", d);
	}
	cv::Scalar tempVal = cv::mean(matDist);
	double myMAtMean = tempVal.val[0];

	double minV, maxV;
	int minI[2];
	int maxI[2];
	cv::minMaxIdx(matDist, &minV, &maxV, minI, maxI);
	//printf("%f,%f,%d,%d,%f\n", minV, maxV, minI[1], maxI[1], myMAtMean);
	//
	cv::Scalar  mean, stddev;
	cv::meanStdDev(matDist, mean, stddev);
	//std::cout << "mean" << mean << std::endl;
	//std::cout << "stdev" << stddev << std::endl;

	return myMAtMean;
}

std::vector<cv::Vec2f> shelfLines(const cv::Mat & img_scene) {

	cv::Mat			temp;
	cv::Mat			img_object_gray;
	cv::cvtColor(img_scene, img_object_gray, CV_BGR2GRAY);
	const int		N = 5;
	const double	C = 2;

	// Calculate a mean over a NxN window and subtracts C from the mean. This is the threshold level for every pixel.
	cv::adaptiveThreshold(img_object_gray, temp, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, N, C);
	// Since we're interested in the cross-lines(shelf on the counter), and they are dark, we invert the image
	cv::bitwise_not(temp, temp);
	//
	// Focus on horizontal lines(shelf on the counter)
	//cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
	cv::Mat kernel_h = (cv::Mat_<uchar>(3, 3) << 0, 0, 0, 1, 1, 1, 0, 0, 0);
	//cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 0, 1, 0, 0, 1, 0);
	//cv::erode(temp, temp, kernel_h);
	cv::erode(temp, temp, kernel_h);
	//cv::erode(temp, temp, kernel_h);
	//cv::dilate(temp, temp, kernel_h);
	cv::dilate(temp, temp, kernel_h);
	//cv::dilate(temp, temp, kernel_h);


	//
	// Find contours
	std::vector<std::vector<cv::Point>>		contours;
	std::vector<cv::Vec4i>					hierarchy;
	cv::findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
	//
	cv::Mat mask = cv::Mat::zeros(temp.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		cv::Rect rect = cv::boundingRect(contours[i]);
		if (area >(temp.cols * 0.5)
			//&&  rect.width / rect.height > 10
			//&& rect.width > temp.cols / 2
			) {
			cv::drawContours(mask, contours, i, cv::Scalar(255), CV_FILLED, 8);
		}
	}
	// DO not use morphology here more because angled tests do not passed them

	// Apply y-gradient to obtain only (semi)horizontal lines from contours
	cv::Mat		grad_y;
	cv::Sobel(mask, grad_y, CV_8U, 0, 1, 1);
	cv::convertScaleAbs(grad_y, mask);


	// Detect lines
	std::vector<cv::Vec2f>		lines;
	cv::HoughLines(mask, lines, 1, CV_PI / 360.0, 250);

	cv::Mat sceneLines = img_scene.clone() * 0.4;
	for (int i = 0;i<lines.size();i++) draw::Lines(lines[i], sceneLines, CV_RGB(0, 255, 255), 1);

	// Merging lines
	std::vector<cv::Vec2f> linesMerged = lines;
	mergeRelatedLines(linesMerged, img_scene, 20);

	// Estimate average period and merge them according to estimated average period
	std::vector<double> linesH;
	for (int i = 0; i < linesMerged.size(); ++i) linesH.push_back(linesMerged[i][0]);
	double linesPeriod = findPeriod(linesH);
	//printf("linesPeriod %f\n", linesPeriod);
	mergeRelatedLines(linesMerged, img_scene, linesPeriod);
	for (int i = 0;i<linesMerged.size();i++) draw::Lines(linesMerged[i], sceneLines, CV_RGB(255, 0, 0), 3);
	linesPeriod = findPeriod(linesH);

	////cv::imshow("temp", temp);
	//cv::imshow("mask", mask);
	//cv::imshow("sceneLines", sceneLines);

	return linesMerged;
}