#ifndef ARPIPELINE_HPP
#define ARPIPELINE_HPP

////////////////////////////////////////////////////////////////////
// File includes:
#include "PatternDetector.hpp"
#include "CameraCalibration.hpp"
#include "GeometryTypes.hpp"

class ARPipeline
{
    struct PatternEntity {
        Pattern             m_pattern;
        PatternTrackingInfo m_patternInfo;
    };

public:
  ARPipeline(const std::vector<cv::Mat>& patternImages, const CameraCalibration& calibration);

  bool processFrame(const cv::Mat& inputFrame);

  /*
  * @brief: Return number of patterns which have been prepared for searching by this object of class.
  */
  const size_t                  getPatternsCount() const { return m_patternEntities.size(); }
  const bool                    isPatternFound(const size_t index) const { if (index < 0 || index >= m_patternEntities.size()) return false; return m_patternEntities[index].m_patternInfo.homographyFound; }
  const PatternTrackingInfo &   getPatternInfo(const size_t index) const { return m_patternEntities[index].m_patternInfo; }

  cv::Ptr<PatternDetector>      m_patternDetector;
private:

private:
  CameraCalibration             m_calibration;
  std::vector<PatternEntity>    m_patternEntities;
};

#endif