#include "VideoProvider.h"


VideoProvider::VideoProvider(const std::string & videoPath) {
    bool successed = m_cap.open(videoPath);
    if (!m_cap.isOpened()) {
        std::cout << "cannot read video: " << videoPath << std::endl;
    }
}

const int VideoProvider::getFrame(cv::Mat & img) const {
    bool success = m_cap.read(img);

    if (m_cap.isOpened() && success == false) {
        // Loop video
        std::cout << "Source video was restarted" << std::endl;
        m_cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    }

    return success ? 0 : -1;
}