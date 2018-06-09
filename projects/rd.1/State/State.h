#pragma once

#include <Provider/Provider.h>

class State {
private:
    cv::Ptr<Provider>   m_src_scene_provider;
    std::string         m_resultWindowName;
public:

    State(const std::string & WindowName);

    const int Initialize(cv::Ptr<Provider> src_scene_provider);

    const int getSceneFrame(cv::Mat & img) const;

    void showImage(const cv::Mat & img) const;

    const std::string wndName() const { return m_resultWindowName; }
};