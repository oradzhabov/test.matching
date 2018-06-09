#include "State.h"

State::State(const std::string & resultWindowName)
:m_resultWindowName(resultWindowName) {

    /// Create Window
    cv::namedWindow(m_resultWindowName, 1);
}

const int State::Initialize(cv::Ptr<Provider> src_scene_provider) {
    m_src_scene_provider = src_scene_provider;
    return 0;
}

const int State::getSceneFrame(cv::Mat & img) const {
    if (m_src_scene_provider)
        return m_src_scene_provider->getFrame(img);

    return -1;
}

void State::showImage(const cv::Mat & img) const {
    cv::imshow(m_resultWindowName, img);
}