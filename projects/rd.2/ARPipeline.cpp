#include "ARPipeline.hpp"


ARPipeline::ARPipeline(const std::vector<cv::Mat>& patternImages, const CameraCalibration& calibration)
  : m_calibration(calibration), m_patternDetector(PatternDetector::CreateAKAZE())
{
    for (size_t i = 0; i < patternImages.size(); i++) {
        PatternEntity patternEntity;

        PatternDetector::buildPatternFromImage(m_patternDetector, patternImages[i], patternEntity.m_pattern);

        m_patternEntities.push_back(patternEntity);
    }

    if (m_patternEntities.empty())
        std::cout << "Attention - No one pattern images had not been loaded" << std::endl;
}

bool ARPipeline::processFrame(const cv::Mat& inputFrame) {

    // Convert input image to gray and extract features from it
    if (m_patternDetector->extractFeatures(inputFrame) == false)
        return false;

    bool anyPatternFound = false;
    for (size_t i = 0; i < m_patternEntities.size(); i++) {

        m_patternDetector->train(m_patternEntities[i].m_pattern);

        if (m_patternDetector->findPattern(inputFrame, m_patternEntities[i].m_patternInfo)) {
            m_patternEntities[i].m_patternInfo.computePose(m_patternEntities[i].m_pattern, m_calibration);
            anyPatternFound = true;
        }
    }

    return anyPatternFound;
}

