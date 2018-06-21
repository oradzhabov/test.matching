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
#include "PatternDetector.hpp"
#include "DebugHelpers.hpp"

////////////////////////////////////////////////////////////////////
// Standard includes:
#include <cmath>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <cassert>

PatternDetector::PatternDetector(cv::Ptr<cv::FeatureDetector> detector, 
    cv::Ptr<cv::DescriptorMatcher> matcher, 
    bool ratioTest)
    : m_detector(detector)
    , m_matcher(matcher)
    , enableRatioTest(ratioTest)
    , enableHomographyRefinement(true)
    , homographyReprojectionThreshold(3)
{
}


void PatternDetector::train(const Pattern& pattern)
{
    // Store the pattern object
    m_pattern = pattern;

    // API of cv::DescriptorMatcher is somewhat tricky
    // First we clear old train data:
    m_matcher->clear();

    // Then we add vector of descriptors (each descriptors matrix describe one image). 
    // This allows us to perform search across multiple images:
    std::vector<cv::Mat> descriptors(1);
    descriptors[0] = pattern.descriptors.clone(); 
    m_matcher->add(descriptors);

    // After adding train data perform actual train:
    m_matcher->train();
}

void PatternDetector::buildPatternFromImage(const PatternDetector * detector, const cv::Mat& image, Pattern& pattern)
{
    // Store original image in pattern structure
    pattern.size = cv::Size(image.cols, image.rows);
    pattern.frame = image.clone();
    PatternDetector::getGray(image, pattern.grayImg);
    PatternDetector::getEdges(pattern.grayImg, pattern.grayImg);
    
    // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
    pattern.points2d.resize(4);
    pattern.points3d.resize(4);

    // Image dimensions
    const float w = static_cast<float>(image.cols);
    const float h = static_cast<float>(image.rows);

    // Normalized dimensions:
    const float maxSize = std::max(w,h);
    const float unitW = w / maxSize;
    const float unitH = h / maxSize;

    // ATTENTION: Direction important because uses in point insertion algs
    pattern.points2d[0] = cv::Point2f(0,0);
    pattern.points2d[1] = cv::Point2f(w,0);
    pattern.points2d[2] = cv::Point2f(w,h);
    pattern.points2d[3] = cv::Point2f(0,h);

    pattern.points3d[0] = cv::Point3f(-unitW, -unitH, 0);
    pattern.points3d[1] = cv::Point3f( unitW, -unitH, 0);
    pattern.points3d[2] = cv::Point3f( unitW,  unitH, 0);
    pattern.points3d[3] = cv::Point3f(-unitW,  unitH, 0);

    PatternDetector::extractFeatures(detector->m_detector, pattern.grayImg, pattern.keypoints, pattern.descriptors);
}



bool PatternDetector::findPattern(const cv::Mat& image, PatternTrackingInfo& info)
{
    // Initialize result
    info.homographyFound = false;

   
    // Get matches with current pattern
    std::vector<cv::DMatch>     matches;
    getMatches(m_queryDescriptors, matches);

#if _DEBUG
    cv::showAndSave("Raw matches", getMatchesImage(image, m_pattern.frame, m_queryKeypoints, m_pattern.keypoints, matches, 100));
#endif

#if _DEBUG
    cv::Mat                     tmp = image.clone();
#endif
    cv::Mat                     roughHomography;
    std::vector<cv::DMatch>     refinedMatches;

    // Find homography transformation and detect good matches
    info.homographyFound = refineMatchesWithHomography(
        m_queryKeypoints, 
        m_pattern.keypoints, 
        homographyReprojectionThreshold, 
        matches,
        CV_FM_RANSAC, // RANSAC good for rough estmation when lot of keypoints with error comes
		roughHomography);

    if (info.homographyFound)
    {
#if _DEBUG
        cv::showAndSave("Refined matches using RANSAC", getMatchesImage(image, m_pattern.frame, m_queryKeypoints, m_pattern.keypoints, matches, 100));
#endif
        // If homography refinement enabled improve found transformation
        if (enableHomographyRefinement)
        {
			cv::Mat                   warpedImg;

            // Warp image using found homography
            cv::warpPerspective(m_grayImg, warpedImg, roughHomography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);

            // Get refined matches:
            std::vector<cv::KeyPoint>	warpedKeypoints;
			cv::Mat						warpedDescriptors;

            // Detect features on warped image
            PatternDetector::extractFeatures(m_detector, warpedImg, warpedKeypoints, warpedDescriptors);

            // Match with pattern
            getMatches(warpedDescriptors, refinedMatches);

            // Estimate new refinement homography
            info.homographyFound = refineMatchesWithHomography(
                warpedKeypoints, 
                m_pattern.keypoints, 
                homographyReprojectionThreshold,
                refinedMatches, 
                CV_FM_LMEDS, // LMEDS good for preciss estmation
                m_refinedHomography);

            if (!info.homographyFound || m_refinedHomography.empty())
                return info.homographyFound;

#if _DEBUG
            cv::showAndSave("MatchesWithWarpedPose", getMatchesImage(warpedImg, m_pattern.grayImg, warpedKeypoints, m_pattern.keypoints, refinedMatches, 100));
#endif
            // Get a result homography as result of matrix product of refined and rough homographies:
            info.homography = roughHomography * m_refinedHomography;

#if _DEBUG
            // Warp image using found refined homography
            cv::Mat warpedRefinedImg;
            cv::warpPerspective(m_grayImg, warpedRefinedImg, info.homography, m_pattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
            cv::showAndSave("Warped Refined image", warpedRefinedImg);

            // Transform contour with rough homography
            cv::perspectiveTransform(m_pattern.points2d, info.points2d, roughHomography);
            info.draw2dContour(tmp, CV_RGB(0,200,0));
#endif

            // Transform contour with precise homography
            cv::perspectiveTransform(m_pattern.points2d, info.points2d, info.homography);
#if _DEBUG
            info.draw2dContour(tmp, CV_RGB(200,0,0));
#endif
        }
        else
        {
            info.homography = roughHomography;

            // Transform contour with rough homography
            cv::perspectiveTransform(m_pattern.points2d, info.points2d, roughHomography);
#if _DEBUG
            info.draw2dContour(tmp, CV_RGB(0,200,0));
#endif
        }
    }

#if _DEBUG
    if (1)
    {
        cv::showAndSave("Final matches", getMatchesImage(tmp, m_pattern.frame, m_queryKeypoints, m_pattern.keypoints, matches, 100));
    }
    std::cout << "Features:" << std::setw(4) << m_queryKeypoints.size() << " Matches: " << std::setw(4) << matches.size();
    if (enableHomographyRefinement)
        std::cout << " Refined Matches: " << std::setw(4) << refinedMatches.size();
    std::cout << std::endl;
#endif

    // Clear found result to prevent remark them later in pipeline
    if (info.homographyFound && m_queryKeypoints.size() > 0) {
        //
        // Filter keypoints(/w descriptors) which were just recognized on the frame
        // They should be avoided from futher matching when next patterns tries to be recognized
        //
        std::vector<size_t> indForErase;
        for (int i = 0; i < m_queryKeypoints.size(); i++) {
            if (cv::pointPolygonTest(info.points2d, m_queryKeypoints[i].pt, false) > 0)
                indForErase.push_back(i);
        }
        std::vector<cv::KeyPoint>   queryKeypoints;
        cv::Mat                     queryDescriptors(0, m_queryDescriptors.cols, m_queryDescriptors.type());
        for (int i = 0; i < m_queryKeypoints.size(); i++) {
            if (std::find(indForErase.begin(), indForErase.end(), i) == indForErase.end()) {
                queryKeypoints.push_back(m_queryKeypoints[i]);
                queryDescriptors.push_back(m_queryDescriptors.row(i));
            }
        }
        m_queryKeypoints = queryKeypoints;
        m_queryDescriptors = queryDescriptors;
    }
    return info.homographyFound;
}

void PatternDetector::getGray(const cv::Mat& image, cv::Mat& gray) {
    assert(!image.empty());

    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

void PatternDetector::getEdges(const cv::Mat& gray, cv::Mat& edges) {
    assert(!gray.empty());
    assert(gray.channels() == 1);

    /// Generate grad_x and grad_y
    const int       ddepth = CV_16S;
    cv::Mat         grad_x, grad_y;
    cv::Mat         abs_grad_x, abs_grad_y;

    /// Gradient X
    cv::Sobel(gray, grad_x, ddepth, 1, 0);

    /// Gradient Y
    cv::Sobel(gray, grad_y, ddepth, 0, 1);

    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
}

bool PatternDetector::extractFeatures(const cv::Mat& image) {
    assert(!image.empty());

    // Convert input image to gray
    PatternDetector::getGray(image, m_grayImg);
    // Convert gray to edges
    PatternDetector::getEdges(m_grayImg, m_grayImg);

	// Extract feature points from input image
	return PatternDetector::extractFeatures(m_detector, m_grayImg, m_queryKeypoints, m_queryDescriptors);
}

bool PatternDetector::extractFeatures(cv::Ptr<cv::FeatureDetector> detector, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    assert(!image.empty());
    assert(image.channels() == 1);

    detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    if (keypoints.empty())
        return false;

    return true;
}

void PatternDetector::getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches, const float & maxDistance, const float & minRatio)
{
    matches.clear();

    if (enableRatioTest)
    {
        m_knnMatches.clear();

        // KNN match will return 2 nearest matches for each query descriptor
        m_matcher->knnMatch(queryDescriptors, m_knnMatches, 2);

        for (size_t i=0; i<m_knnMatches.size(); i++)
        {
            const cv::DMatch& bestMatch   = m_knnMatches[i][0];
            const cv::DMatch& betterMatch = m_knnMatches[i][1];

            // To avoid NaN's when best match has zero distance we will use inversed ratio. 
            float distanceRatio = bestMatch.distance / betterMatch.distance;
            
            // Pass only matches where distance ratio between 
            // nearest matches is greater than 1.5 (distinct criteria)
            if (distanceRatio < minRatio && bestMatch.distance <= maxDistance)
            {
                matches.push_back(bestMatch);
            }
        }
    }
    else
    {
        // Perform regular match
        m_matcher->match(queryDescriptors, matches);
    }
}

bool PatternDetector::refineMatchesWithHomography
    (
    const std::vector<cv::KeyPoint>& queryKeypoints,
    const std::vector<cv::KeyPoint>& trainKeypoints, 
    float reprojectionThreshold,
    std::vector<cv::DMatch>& matches,
    const int method,
    cv::Mat& homography
    )
{
    const int minNumberMatchesAllowed = 8;

    if (matches.size() < minNumberMatchesAllowed)
        return false;

    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());

    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, 
                                    dstPoints, 
                                    method, 
                                    reprojectionThreshold, 
                                    inliersMask);

    std::vector<cv::DMatch> inliers;
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }

    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}
