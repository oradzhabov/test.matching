#pragma once
#include <main.h>

class KPDetector {
private:
    cv::Mat                         m_img;
    cv::Ptr<cv::Feature2D>          m_detector;
    std::vector<cv::KeyPoint>		m_keypoints;
    cv::Mat							m_descriptors;
    std::vector<cv::Point2f>        m_matched_pts;
public:
    KPDetector();

    const int   detect(const cv::Mat & img);

    const bool haveDetection() const { return !m_keypoints.empty(); };

    int match(KPDetector & rhs);
};