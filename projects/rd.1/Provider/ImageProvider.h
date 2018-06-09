#pragma once
#include "Provider.h"

class ImageProvider : public Provider {
private:
    cv::Mat         m_img_scene;
public:
    ImageProvider(const std::string & imgPath);

private:
    virtual const int getFrame(cv::Mat & img) const;
};