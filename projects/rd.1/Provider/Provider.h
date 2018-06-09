#pragma once
#include <main.h>

class Provider {
public:

    virtual const int getFrame(cv::Mat & img) const = 0;

public:
    static cv::Ptr<Provider> CreateImageProvider(const std::string & imgPath);
};