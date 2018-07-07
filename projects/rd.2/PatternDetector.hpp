#ifndef EXAMPLE_MARKERLESS_AR_PATTERNDETECTOR_HPP
#define EXAMPLE_MARKERLESS_AR_PATTERNDETECTOR_HPP

////////////////////////////////////////////////////////////////////
// File includes:
#include "Pattern.hpp"

#include <opencv2/opencv.hpp>
#include <OppColorDescriptorExtractor.h>

class PatternDetector
{
public:
    /**
     * Initialize a pattern detector with specified feature detector, descriptor extraction and matching algorithm
     */
    PatternDetector (cv::Ptr<cv::FeatureDetector>        detector,
                    cv::Ptr<cv::DescriptorExtractor>    extractor,
                    cv::Ptr<cv::DescriptorMatcher>      matcher,
                    bool enableRatioTest);

    static cv::Ptr<PatternDetector>    CreateAKAZE();
    static cv::Ptr<PatternDetector>    CreateBRISK();

    bool extractFeatures(const cv::Mat& image, const cv::Mat & mask = cv::Mat());
    /**
    * 
    */
    void train(const Pattern * pattern);

    static int ratioTest(std::vector<std::vector<cv::DMatch> > & matches, const float & minRatio);
    static void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches);
    static void horizontalTest(const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> & matches, const int imgWidth);

    /**
    * Initialize Pattern structure from the input image.
    * This function finds the feature points and extract descriptors for them.
    */
    static void buildPatternFromImage(const cv::Ptr<PatternDetector> detector, const cv::Mat& image, Pattern& pattern);

    /**
    * Tries to find a @pattern object on given @image. 
    * The function returns true if succeeded and store the result (pattern 2d location, homography) in @info.
    */
    bool findPattern(const cv::Mat& image, PatternTrackingInfo& info);

    bool enableRatioTest;
    bool enableHomographyRefinement;
    float homographyReprojectionThreshold;

protected:

    static bool extractFeatures(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, const cv::Mat& imageGray, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat & mask = cv::Mat());

    void getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches, const float & minRatio = 1.f / 1.2f, const float & maxDistance = std::numeric_limits<float>::max());
    void getRatiotedAndSortedMatches(const cv::Mat& queryDescriptors, std::list<cv::DMatch>& matches);
    /**
    * Get the gray image from the input image.
    * Function performs necessary color conversion if necessary
    * Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
    */
    static void getGray(const cv::Mat& image, cv::Mat& grayBlured, cv::Mat& gray);
    static void getEdges(const cv::Mat& gray, cv::Mat& edges);

    /**
    * 
    */
    static bool refineMatchesWithHomography(
        const std::vector<cv::KeyPoint>& queryKeypoints, 
        const std::vector<cv::KeyPoint>& trainKeypoints, 
        float reprojectionThreshold,
        std::vector<cv::DMatch>& matches, 
        const int method,
        cv::Mat& homography,
        const int minNumberMatchesAllowed);

private:
    std::vector<cv::KeyPoint> m_queryKeypoints;
    cv::Mat                   m_queryDescriptors;
    std::vector< std::vector<cv::DMatch> > m_knnMatches;

    cv::Mat                   m_grayBluredImg;
    cv::Mat                   m_grayImg;

    const Pattern *                  m_pPattern;
    cv::Ptr<cv::FeatureDetector>     m_detector;
    cv::Ptr<cv::DescriptorExtractor>    m_extractor;
    cv::Ptr<cv::DescriptorMatcher>   m_matcher;
};

#endif