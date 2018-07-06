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
    int maxEdgeSize = 0;
    for (size_t i = 0; i < patternImages.size(); i++) {

        PatternEntity patternEntity;

        PatternDetector::buildPatternFromImage(&m_patternDetector, patternImages[i], patternEntity.m_pattern);

        maxEdgeSize = std::max<int>(maxEdgeSize, patternEntity.m_pattern.size.width);
        maxEdgeSize = std::max<int>(maxEdgeSize, patternEntity.m_pattern.size.height);

        m_patternEntities.push_back(patternEntity);
    }

    // Found scale factors to uniform all patterns by maximum
    for (size_t i = 0; i < m_patternEntities.size(); i++) {
        int ptrnMaxSize = std::max<int>(m_patternEntities[i].m_pattern.size.width, m_patternEntities[i].m_pattern.size.height);
        const float scaleUniform = static_cast<float>(maxEdgeSize) / static_cast<float>(ptrnMaxSize);
        for (size_t j = 0; j < 4; j++) {
            m_patternEntities[i].m_pattern.points3d[j] /= scaleUniform;
        }
    }

    if (m_patternEntities.empty())
        std::cout << "Attention - No one pattern images had not been loaded" << std::endl;
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

const PatternTrackingInfo& ARPipeline::getPatternInfo(const size_t index) const {
    return m_patternEntities[index].m_patternInfo;
}