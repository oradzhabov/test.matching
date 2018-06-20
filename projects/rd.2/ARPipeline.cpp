/*****************************************************************************
*   Markerless AR desktop application.
******************************************************************************
*   by Khvedchenia Ievgen, 5th Dec 2012
*   http://computer-vision-talks.com
******************************************************************************
*   Ch3 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

////////////////////////////////////////////////////////////////////
// File includes:
#include "ARPipeline.hpp"

ARPipeline::ARPipeline(const std::vector<cv::Mat>& patternImages, const CameraCalibration& calibration)
  : m_calibration(calibration)
{
    for (size_t i = 0; i < patternImages.size(); i++) {

        PatternEntity patternEntity;

        PatternDetector::buildPatternFromImage(&m_patternDetector, patternImages[i], patternEntity.m_pattern);

        m_patternEntities.push_back(patternEntity);
    }
}

bool ARPipeline::processFrame(const cv::Mat& inputFrame) {

    // Convert input image to gray and extract features from it
    if (m_patternDetector.extractFeatures(inputFrame) == false)
        return false;

    bool anyPatternFound = false;
    for (size_t i = 0; i < m_patternEntities.size(); i++) {

        m_patternDetector.train(m_patternEntities[i].m_pattern);

        if (m_patternDetector.findPattern(inputFrame, m_patternEntities[i].m_patternInfo)) {
            m_patternEntities[i].m_patternInfo.computePose(m_patternEntities[i].m_pattern, m_calibration);
            anyPatternFound = true;
        }
    }

    return anyPatternFound;
}

const size_t ARPipeline::getPatternsCount() const {
    return m_patternEntities.size();
}

const bool ARPipeline::isPatternFound(const size_t index) const {
    if (index < 0 || index >= m_patternEntities.size()) return false;

    return m_patternEntities[index].m_patternInfo.homographyFound;
}

const Transformation& ARPipeline::getPatternLocation(const size_t index) const {
    return m_patternEntities[index].m_patternInfo.pose3d;
}