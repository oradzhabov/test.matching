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
  const size_t          getPatternsCount() const;
  const bool            isPatternFound(const size_t index) const;
  const PatternTrackingInfo& getPatternInfo(const size_t index) const;

  PatternDetector     m_patternDetector;
private:

private:
  CameraCalibration             m_calibration;
  std::vector<PatternEntity>    m_patternEntities;
};

#endif