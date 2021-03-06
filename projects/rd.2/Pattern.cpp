#include "Pattern.hpp"

void PatternTrackingInfo::computePose(const Pattern& pattern, const CameraCalibration& calibration)
{
  cv::Mat Rvec;
  cv::Mat_<float> Tvec;

  cv::solvePnP(pattern.points3d, points2d, calibration.getIntrinsic(), calibration.getDistorsion(),raux,taux, useExtrinsicGuess, cv::SOLVEPNP_ITERATIVE);
  raux.convertTo(Rvec,CV_32F);
  taux.convertTo(Tvec ,CV_32F);

  cv::Mat_<float> rotMat(3,3); 
  cv::Rodrigues(Rvec, rotMat);

  // Copy to transformation matrix
  for (int col=0; col<3; col++)
  {
    for (int row=0; row<3; row++)
    {        
        pose3d.r().mat[row][col] = rotMat(row,col); // Copy rotation component
    }
    pose3d.t().data[col] = Tvec(col); // Copy translation component
  }

  // Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
  pose3d = pose3d.getInverted();
}

void PatternTrackingInfo::draw2dContour(cv::Mat& image, cv::Scalar color) const
{
  for (size_t i = 0; i < points2d.size(); i++) cv::line(image, points2d[i], points2d[ (i+1) % points2d.size() ], color, 2, CV_AA);
}

void PatternTrackingInfo::fill2dContour(cv::Mat& image, cv::Scalar color) const {

    assert(!image.empty());

    std::vector<cv::Point> pts;
    
    for (size_t i = 0; i < points2d.size(); i++)
        pts.push_back(cv::Point(static_cast<int>(points2d[i].x), static_cast<int>(points2d[i].y)));


    if (pts.empty())
        return;

    const cv::Point* ppt[1] = { &pts[0] };
    int npt[] = { static_cast<int>(points2d.size()) };

    cv::fillPoly(image,
        ppt,
        npt,
        1,
        color,
        cv::LINE_8);
}
