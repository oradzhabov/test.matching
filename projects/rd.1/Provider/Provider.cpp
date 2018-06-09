#include "Provider.h"
#include "ImageProvider.h"

cv::Ptr<Provider> Provider::CreateImageProvider(const std::string & imgPath) {
    cv::Ptr<Provider> ptr = new ImageProvider(imgPath);
    return ptr;
}