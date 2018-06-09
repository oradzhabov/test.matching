#include "Provider.h"
#include "ImageProvider.h"
#include "VideoProvider.h"

cv::Ptr<Provider> Provider::CreateImageProvider(const std::string & imgPath) {
    cv::Ptr<Provider> ptr = new ImageProvider(imgPath);
    return ptr;
}

cv::Ptr<Provider> Provider::CreateVideoProvider(const std::string & videoPath) {
    cv::Ptr<Provider> ptr = new VideoProvider(videoPath);
    return ptr;
}