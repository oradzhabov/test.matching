#include "ImageProvider.h"

ImageProvider::ImageProvider(const std::string & imgPath) {
    m_img_scene = cv::imread(imgPath);
}

const int ImageProvider::getFrame(cv::Mat & img) const {
    if (m_img_scene.empty())
        return -1;

    img = m_img_scene.clone(); // todo: really clone()?
    return 0;
}