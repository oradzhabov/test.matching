#pragma once
#include "Provider.h"

class VideoProvider : public Provider {
private:
    mutable cv::VideoCapture        m_cap;
public:
    VideoProvider(const std::string & videoPath);

private:
    virtual const int getFrame(cv::Mat & img) const;
};