#ifndef EXAMPLE_MARKERLESS_AR_PATTERN_HPP
#define EXAMPLE_MARKERLESS_AR_PATTERN_HPP

#include "GeometryTypes.hpp"
#include "CameraCalibration.hpp"

#include <opencv2/opencv.hpp>

/**
 * Store the image data and computed descriptors of target pattern
 */
struct Pattern
{
  cv::Size                  size;
  cv::Mat                   frame;
  cv::Mat                   grayImg;

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat                   descriptors;

  std::vector<cv::Point2f>  points2d;
  std::vector<cv::Point3f>  points3d;
};

/**
 * Intermediate pattern tracking info structure
 */
struct PatternTrackingInfo
{
  bool                      homographyFound;
  cv::Mat                   homography;
  std::vector<cv::Point2f>  points2d;
  Transformation            pose3d;
  bool                      useExtrinsicGuess; ///< if true the function solvePnP() WITH ITERATIVE METHOD ONLY uses the provided rvec and tvec values as initial approximations of the rotation and translation vectors
  cv::Mat                   raux;
  cv::Mat                   taux;


  PatternTrackingInfo() :homographyFound(false), useExtrinsicGuess(false) {}

  void draw2dContour(cv::Mat& image, cv::Scalar color) const;
  void fill2dContour(cv::Mat& image, cv::Scalar color) const;

  /**
   * Compute pattern pose using PnP algorithm
   */
  void computePose(const Pattern& pattern, const CameraCalibration& calibration);
};

#endif