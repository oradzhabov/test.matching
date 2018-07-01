#ifndef PORTRAITOBS_HPP
#define PORTRAITOBS_HPP

#include <vector>
#include <Pattern.hpp>

class PortraitObs {
public:
    cv::Mat_<float>                 m_position; // Pattern 3d-position(float) relative CAM position
    std::vector<cv::Mat_<float>>    m_K12; // Transofrmation matrices from current pattern location
public:

    PortraitObs (PatternTrackingInfo &, std::vector<PatternTrackingInfo> &);
};

#endif // PORTRAITOBS_HPP