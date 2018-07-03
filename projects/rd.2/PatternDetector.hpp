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
    PatternDetector
    (
        //cv::Ptr<cv::FeatureDetector>     detector = cv::ORB::create(1000),
        //
        //cv::Ptr<cv::FeatureDetector>        detector = cv::BRISK::create(60),
        //cv::Ptr<cv::FeatureDetector>        extractor = cv::BRISK::create(),
        //cv::Ptr<cv::DescriptorExtractor>    extractor = cv::Ptr<OppColorDescriptorExtractor>( new OppColorDescriptorExtractor(cv::BRISK::create())), //OppColorDeswcriptor does not work with AKAZE
        //
        //cv::Ptr<cv::FeatureDetector>        detector = cv::MSER::create(),
        //cv::Ptr<cv::DescriptorExtractor>        extractor = cv::BRISK::create(),
        //
        // // AKAZE more accurately than BRISK(Binary Robust Invariant Scalable Keypoints), but BRISK faster and good for first approach
        cv::Ptr<cv::FeatureDetector>        detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,1,0.0001f/*found much more points*/,7/*as more as can see more in far from cam. todo: really?*/),
        cv::Ptr<cv::DescriptorExtractor>    extractor = cv::AKAZE::create(),
        //
        // NORM_HAMMING should be used with ORB, BRISK, AKAZE and BRIEF
        // NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description)
        cv::Ptr<cv::DescriptorMatcher>      matcher = new cv::BFMatcher(cv::NORM_HAMMING, false),
        bool enableRatioTest                       = true // ros: ATTENTION: true or here or in second param in BFMatcher. Note: If here, it will drop bad results. Insteads of whether true in BFMatcher
        );

    bool extractFeatures(const cv::Mat& image);
    /**
    * 
    */
    void train(const Pattern& pattern);

    static int ratioTest(std::vector<std::vector<cv::DMatch> > & matches, const float & minRatio);
    static void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches);
    static void horizontalTest(const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> & matches, const int imgWidth);

    /**
    * Initialize Pattern structure from the input image.
    * This function finds the feature points and extract descriptors for them.
    */
    static void buildPatternFromImage(const PatternDetector * detector, const cv::Mat& image, Pattern& pattern);

    /**
    * Tries to find a @pattern object on given @image. 
    * The function returns true if succeeded and store the result (pattern 2d location, homography) in @info.
    */
    bool findPattern(const cv::Mat& image, PatternTrackingInfo& info);

    bool enableRatioTest;
    bool enableHomographyRefinement;
    float homographyReprojectionThreshold;

protected:

    static bool extractFeatures(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, const cv::Mat& imageGray, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    void getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches, const float & minRatio = 1.f / 1.2f, const float & maxDistance = std::numeric_limits<float>::max());

    /**
    * Get the gray image from the input image.
    * Function performs necessary color conversion if necessary
    * Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
    */
    static void getGray(const cv::Mat& image, cv::Mat& gray);
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

    cv::Mat                   m_grayImg;

    Pattern                          m_pattern;
    cv::Ptr<cv::FeatureDetector>     m_detector;
    cv::Ptr<cv::DescriptorExtractor>    m_extractor;
    cv::Ptr<cv::DescriptorMatcher>   m_matcher;
};

#endif