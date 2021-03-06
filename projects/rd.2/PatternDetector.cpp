// todo: maybe good idea - optflow for tracking
// http://answers.opencv.org/question/51749/c-video-stabilization-pipeline-i-certainly-missed-something/

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
    cv::Ptr<cv::DescriptorExtractor> extractor,
    cv::Ptr<cv::DescriptorMatcher> matcher, 
    bool ratioTest)
    : m_detector(detector)
    , m_extractor(extractor)
    , m_matcher(matcher)
    , enableRatioTest(ratioTest)
    , enableHomographyRefinement(true)
    , homographyReprojectionThreshold(3)
{
}

cv::Ptr<PatternDetector> PatternDetector::CreateAKAZE() {
    // NORM_HAMMING should be used with ORB, BRISK, AKAZE and BRIEF
    // NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description)

    // ros: ATTENTION: true or here or in second param in BFMatcher. Note: If here, it will drop bad results. Insteads of whether true in BFMatcher);
    bool enableRatioTest = true;
    cv::Ptr<PatternDetector> ptr = cv::makePtr<PatternDetector>(cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 1, 0.0001f/*found much more points*/, 7/*as more as can see more in far from cam. todo: really?*/),
                                                                cv::AKAZE::create(),
                                                                new cv::BFMatcher(cv::NORM_HAMMING, false),
                                                                enableRatioTest);
    return ptr;
}
cv::Ptr<PatternDetector> PatternDetector::CreateBRISK() {
    // NORM_HAMMING should be used with ORB, BRISK, AKAZE and BRIEF
    // NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description)

    // ros: ATTENTION: true or here or in second param in BFMatcher. Note: If here, it will drop bad results. Insteads of whether true in BFMatcher);
    bool enableRatioTest = true;
    cv::Ptr<PatternDetector> ptr = cv::makePtr<PatternDetector>(cv::BRISK::create(5), // its slow, but the best to see far from cam. About 6k keypoints
                                                                //cv::Ptr<OppColorDescriptorExtractor>( new OppColorDescriptorExtractor(cv::BRISK::create())), //OppColorDeswcriptor does not work with AKAZE
                                                                cv::BRISK::create(),
                                                                new cv::BFMatcher(cv::NORM_HAMMING, false),
                                                                enableRatioTest);
    return ptr;
}


void PatternDetector::train(const Pattern * pattern)
{
    // Store the pattern object
    m_pPattern = pattern;

    // API of cv::DescriptorMatcher is somewhat tricky
    // First we clear old train data:
    m_matcher->clear();

    /*
    // Then we add vector of descriptors (each descriptors matrix describe one image). // todo: we can match several images by one pass? 
    // This allows us to perform search across multiple images:
    std::vector<cv::Mat> descriptors(1);
    descriptors[0] = pattern.descriptors.clone(); 
    m_matcher->add(descriptors);
    */
    m_matcher->add(m_pPattern->descriptors);

    // After adding train data perform actual train:
    m_matcher->train();
}

void PatternDetector::buildPatternFromImage(const cv::Ptr<PatternDetector> detector, const cv::Mat& image, Pattern& pattern)
{
    // Store original image in pattern structure
    pattern.size = cv::Size(image.cols, image.rows);
    pattern.frame = image.clone();
    PatternDetector::getGray(image, pattern.grayBluredImg, pattern.grayImg);
    
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

    PatternDetector::extractFeatures(detector->m_detector, detector->m_extractor, pattern.grayImg, pattern.frame, pattern.keypoints, pattern.descriptors);
}

void PatternDetector::horizontalTest(const std::vector<cv::KeyPoint>& queryKp, const std::vector<cv::KeyPoint>& trainKp, std::vector<cv::DMatch> & matches, const int imgWidth) {

    const double                angleThreshold = CV_PI / 180.0;
    std::vector<cv::DMatch>     result;
    for (size_t i = 0; i < matches.size(); i++) {

        const cv::Point2f & srcPt = trainKp[matches[i].trainIdx].pt;
        const cv::Point2f & dstPt = queryKp[matches[i].queryIdx].pt + cv::Point2f(static_cast<float>(imgWidth), 0.0f);
        //
        const cv::Point2f   delta = dstPt - srcPt;
        const double        angDegree = atan2(delta.y, delta.x);

        if (fabs(angDegree) < angleThreshold) result.push_back(matches[i]);
    }
    matches = result;
}

bool PatternDetector::findPattern(const cv::Mat& image, PatternTrackingInfo& info) {

    // Store last homography searching status
    const bool didHomographyFound = info.homographyFound;

    // Initialize result
    info.homographyFound = false;
    info.useExtrinsicGuess = false;

    // During iterating between patterns, descriptors could be cleaned. So check it here
    if (m_queryDescriptors.empty())
        return false;

    cv::Mat                     roughHomography;
    std::vector<cv::DMatch>     refinedMatches;
    std::vector<cv::DMatch>     matches;
#if _DEBUG
    cv::Mat                     tmp = image.clone();
#endif

    if (!didHomographyFound || !enableHomographyRefinement) { // Prev frame was not successfull or we not use refinement at all. So heve we need found rough homography

        // Get matches with current pattern
        getMatches(m_queryDescriptors, matches);

#if _DEBUG
        cv::showAndSave("Raw matches", getMatchesImage(m_grayImg, m_pPattern->grayImg, m_queryKeypoints, m_pPattern->keypoints, matches, matches.size()));
        //cv::showAndSave("Raw matches", getMatchesImage(image, m_pattern.frame, m_queryKeypoints, m_pattern.keypoints, matches, 100));
#endif
        info.homographyFound = refineMatchesWithHomography(
            m_queryKeypoints,
            m_pPattern->keypoints,
            homographyReprojectionThreshold,
            matches,
            CV_FM_RANSAC, // RANSAC good for rough estmation when lot of keypoints with error comes
            roughHomography, 20); // as less this count as faster algorithm and vice versa - as bigger - as more accurately
#if _DEBUG
        cv::showAndSave("Refined matches using RANSAC", getMatchesImage(image, m_pPattern->frame, m_queryKeypoints, m_pPattern->keypoints, matches, matches.size()));
#endif
    }
    else {
        info.homographyFound = true;
        roughHomography = info.homography;
    }
        
    if (info.homographyFound)
    {
        // If homography refinement enabled improve found transformation
        if (enableHomographyRefinement)
        {
            const int               warpFlags = cv::WARP_INVERSE_MAP | cv::INTER_LINEAR;
            cv::Mat                 warpedImg;
            cv::Mat                 warpedImgBGR;
            cv::Mat                 refinedHomography;

            // Warp image using found homography
            cv::warpPerspective(m_grayImg, warpedImg, roughHomography, m_pPattern->size, warpFlags);
            // Prepare colored warp if necessary
            if (m_extractor->getDefaultName() == OppColorDescriptorExtractor::DefaultName)
                cv::warpPerspective(image, warpedImgBGR, roughHomography, m_pPattern->size, warpFlags);

            // Get refined matches:
            std::vector<cv::KeyPoint>	warpedKeypoints;
            cv::Mat						warpedDescriptors;

            // Detect features on warped image
            info.homographyFound = PatternDetector::extractFeatures(m_detector, m_extractor, warpedImg, warpedImgBGR, warpedKeypoints, warpedDescriptors);
            if (!info.homographyFound)
                return false;

            // Match with pattern
            getMatches(warpedDescriptors, refinedMatches, 1.0f / 1.4f);

            // Warped matching NEEDS to be horizontal and collinear
            PatternDetector::horizontalTest(warpedKeypoints, m_pPattern->keypoints, refinedMatches, warpedImg.cols);

            // Estimate new refinement homography
            info.homographyFound = refineMatchesWithHomography(
                warpedKeypoints, 
                m_pPattern->keypoints,
                homographyReprojectionThreshold,
                refinedMatches, 
                CV_FM_LMEDS, // LMEDS good for preciss estmation
                refinedHomography, 4);

            if (info.homographyFound && !refinedHomography.empty()) {
#if _DEBUG
                cv::showAndSave("MatchesWithWarpedPose", getMatchesImage(warpedImg, m_pPattern->grayImg, warpedKeypoints, m_pPattern->keypoints, refinedMatches, 100));
#endif
                // Get a result homography as result of matrix product of refined and rough homographies:
                info.homography = roughHomography * refinedHomography;

#if _DEBUG
                // Warp image using found refined homography
                cv::Mat warpedRefinedImg;
                cv::warpPerspective(m_grayImg, warpedRefinedImg, info.homography, m_pPattern->size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);
                cv::showAndSave("Warped Refined image", warpedRefinedImg);

                // Transform contour with rough homography
                cv::perspectiveTransform(m_pPattern->points2d, info.points2d, roughHomography);
                info.draw2dContour(tmp, CV_RGB(0, 200, 0));
#endif

                // Transform contour with precise homography
                cv::perspectiveTransform(m_pPattern->points2d, info.points2d, info.homography);
#if _DEBUG
                info.draw2dContour(tmp, CV_RGB(200, 0, 0));
#endif
            }
        }
        else
        {
            info.homography = roughHomography;

            // Transform contour with rough homography
            cv::perspectiveTransform(m_pPattern->points2d, info.points2d, roughHomography);
#if _DEBUG
            info.draw2dContour(tmp, CV_RGB(0,200,0));
#endif
        }
    }

#if _DEBUG
    if (1)
    {
        cv::showAndSave("Final matches", getMatchesImage(tmp, m_pPattern->frame, m_queryKeypoints, m_pPattern->keypoints, matches, 100));
    }
    std::cout << "Features:" << std::setw(4) << m_queryKeypoints.size() << " Matches: " << std::setw(4) << matches.size();
    if (enableHomographyRefinement)
        std::cout << " Refined Matches: " << std::setw(4) << refinedMatches.size();
    std::cout << std::endl;
#endif

    // If previous homography were found correctly
    if (info.homographyFound && didHomographyFound)
        info.useExtrinsicGuess = true;

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

void PatternDetector::getGray(const cv::Mat& image, cv::Mat& grayBlured, cv::Mat& gray) {
    assert(!image.empty());

    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;


    // Sharp image
    // Greatest improved feature detecting/extracting
    // Moreover, the simpliest case: kernel=3 demostrates the best influence
    cv::GaussianBlur(gray, grayBlured, cv::Size(0, 0), 3);
    cv::addWeighted(gray, 1.5, grayBlured, -0.5, 0, gray);
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

bool PatternDetector::extractFeatures(const cv::Mat& image, const cv::Mat & mask) {
    assert(!image.empty());

    // Convert input image to gray
    PatternDetector::getGray(image, m_grayBluredImg, m_grayImg);

	// Extract feature points from input image
    bool result = PatternDetector::extractFeatures(m_detector, m_extractor, m_grayImg, image, m_queryKeypoints, m_queryDescriptors, mask);

#if _DEBUG2
    // Draw keypoints
    cv::Mat tmp;
    // Draw the keypoints with scale and orientation information
    cv::drawKeypoints(m_grayImg,    // original image
        std::vector<cv::KeyPoint>(m_queryKeypoints.begin(), m_queryKeypoints.begin() + std::min<size_t>(m_queryKeypoints.size(), 10000)),			// vector of keypoints
        tmp,				        // the resulting image
        cv::Scalar(0, 255, 0),	    // color of the points
        cv::DrawMatchesFlags::DEFAULT); //drawing flag
    cv::imshow("Keypoints", tmp);
#endif // _DEBUG

    return result;
}

bool PatternDetector::extractFeatures(cv::Ptr<cv::FeatureDetector> detector, cv::Ptr<cv::DescriptorExtractor> extractor, const cv::Mat& imageGray, const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat & mask)
{
    assert(!imageGray.empty());
    assert(imageGray.channels() == 1);

    // Found keypoints
    detector->detect(imageGray, keypoints, mask);

    if (false) { // useful for BRISK detectors only

        // Filter keypoints by quality
        std::vector<cv::KeyPoint> keypointsFiltered;
        for (size_t i = 0; i < keypoints.size(); i++)
            if (keypoints[i].response > 10)
                keypointsFiltered.push_back(keypoints[i]);

        // If filtering have any sense, swap them to result keypoints
        if (!keypointsFiltered.empty())
            keypointsFiltered.swap(keypoints);
    }

    if (keypoints.empty())
        return false;

    // Extract descriptors
    if (extractor->getDefaultName() == OppColorDescriptorExtractor::DefaultName)
        extractor->compute(image, keypoints, descriptors);
    else
        extractor->compute(imageGray, keypoints, descriptors);

    if (descriptors.empty())
        return false;

    return true;
}


int PatternDetector::ratioTest(std::vector<std::vector<cv::DMatch> > & matches, const float & minRatio) {

    int removed = 0;

    // for all matches
    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator) {

        // if 2 NN has been identified
        if (matchIterator->size() > 1) {
            // check distance ratio
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > minRatio) {
                matchIterator->clear(); // remove match
                removed++;
            }
        }
        else
        { // does not have 2 neighbours
            matchIterator->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

void PatternDetector::symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1, const std::vector<std::vector<cv::DMatch> >& matches2, std::vector<cv::DMatch>& symMatches) {

    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1) {

        // ignore deleted matches
        if (matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2) {
            // ignore deleted matches
            if (matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {

                // add symmetrical match
                symMatches.push_back( cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }
}

void PatternDetector::getRatiotedAndSortedMatches(const cv::Mat& queryDescriptors, std::list<cv::DMatch>& matches) {
    const std::vector<cv::Mat>              & trainDescriptors = m_matcher->getTrainDescriptors();
    std::vector<std::vector<cv::DMatch> >   matches12;
    std::vector<std::vector<cv::DMatch> >   matches21;

    matches.clear();

    // Prealloc memory to speed up knnMatch() with working with std::Vector
    matches12.reserve(queryDescriptors.rows);
    matches21.reserve(trainDescriptors[0].rows);

    // From image 1 to image 2
    m_matcher->knnMatch(queryDescriptors, trainDescriptors[0], matches12, 2); // return 2 nearest neighbours

    // From image 2 to image 1
    m_matcher->knnMatch(trainDescriptors[0], queryDescriptors, matches21, 2); // return 2 nearest neighbours

    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1 = matches12.begin(); matchIterator1 != matches12.end(); ++matchIterator1) {
        // ignore
        if (matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2 = matches21.begin(); matchIterator2 != matches21.end(); ++matchIterator2) {
            // ignore
            if (matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {

                float ratio12 = (*matchIterator1)[0].distance / (*matchIterator1)[1].distance;
                float ratio21 = (*matchIterator2)[0].distance / (*matchIterator2)[1].distance;

                cv::DMatch m = cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance);
                m.distance = std::max<float>(ratio12, ratio21); // use distance as maximum ratio
                matches.push_back(m);
                break; // next match in image 1 -> image 2
            }
        }
    }
    // Sort matches by increasing the ratio. First - best, last - worst
    //std::sort(matches.begin(), matches.end(), [](const cv::DMatch & a, const cv::DMatch & b) -> bool { return a.distance < b.distance; });
    matches.sort([](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    /*...
    for (int i = 0; i < symMatches.size(); i++) {
    if (symMatches[i].ratio > minRatio)
    matches;
    }
    */
}

void PatternDetector::getMatches(const cv::Mat& queryDescriptors, std::vector<cv::DMatch>& matches, const float & minRatio, const float & maxDistance)
{
    matches.clear();

    // Validate source data
    if (queryDescriptors.empty())
        return;

    if (enableRatioTest)
    {
        if (false) {
            m_knnMatches.clear();

            // KNN match will return 2 nearest matches for each query descriptor
            m_matcher->knnMatch(queryDescriptors, m_knnMatches, 2);

            for (size_t i = 0; i < m_knnMatches.size(); i++)
            {
                const cv::DMatch& bestMatch = m_knnMatches[i][0];
                const cv::DMatch& betterMatch = m_knnMatches[i][1];

                // To avoid NaN's when best match has zero distance we will use inversed ratio. 
                float distanceRatio = bestMatch.distance / betterMatch.distance;

                // Pass only matches where distance ratio between 
                // nearest matches is greater than 1.5 (distinct criteria)
                if (distanceRatio <= minRatio && bestMatch.distance <= maxDistance)
                {
                    matches.push_back(bestMatch);
                }
            }
        }
        else { // Good because use max filter tests, but bad because works only for 1-train per 1-query
            std::vector<std::vector<cv::DMatch> > matches12, matches21;

            const std::vector<cv::Mat> & trainDescriptors = m_matcher->getTrainDescriptors();

            // Prealloc memory to speed up knnMatch() with working with std::Vector
            matches12.reserve(queryDescriptors.rows);
            matches21.reserve(trainDescriptors[0].rows);


            // From image 1 to image 2
            m_matcher->knnMatch(queryDescriptors, trainDescriptors[0], matches12, 2); // return 2 nearest neighbours

            // From image 2 to image 1
            m_matcher->knnMatch(trainDescriptors[0], queryDescriptors, matches21, 2); // return 2 nearest neighbours

            if (false) { // new implementation
                std::list<std::vector<cv::DMatch> >   matches1f;
                std::list<std::vector<cv::DMatch> >   matches2f;

                // (Ratio test) Filter matches for which NN ratio is > than threshold
                std::copy_if(matches12.begin(), matches12.end(), std::back_inserter(matches1f), [minRatio](std::vector<cv::DMatch> i) { return i.size() > 1 && i[0].distance / i[1].distance <= minRatio; });
                std::copy_if(matches21.begin(), matches21.end(), std::back_inserter(matches2f), [minRatio](std::vector<cv::DMatch> i) { return i.size() > 1 && i[0].distance / i[1].distance <= minRatio; });

                std::list<std::vector<cv::DMatch> >::iterator matches1f_s = matches1f.begin();
                std::list<std::vector<cv::DMatch> >::iterator matches1f_e = matches1f.end();

                if (false) { // optimize
                    // Sort to optimize
                    matches1f.sort([](const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) { return a.front().queryIdx < b.front().queryIdx; });
                    matches2f.sort([](const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) { return a.front().trainIdx < b.front().trainIdx; });

                    // Found bound values
                    int matches1fMin = matches1f.front().front().queryIdx;
                    int matches1fMax = matches1f.back().front().queryIdx;
                    int matches2fMin = matches2f.front().front().trainIdx;
                    int matches2fMax = matches2f.back().front().trainIdx;

                    // Found bound iterators
                    std::vector<cv::DMatch> itt = { cv::DMatch(matches2fMin,0, 0.0f) , cv::DMatch(matches2fMin,0, 0.0f) };
                    matches1f_s = std::lower_bound(matches1f.begin(), matches1f.end(), itt,
                        [matches2fMin](std::vector<cv::DMatch> lhs, std::vector<cv::DMatch> rhs) -> bool { return lhs.front().queryIdx < rhs.front().queryIdx; });
                    if (matches1f_s != matches1f.begin())
                        matches1f_s--;

                    itt = { cv::DMatch(matches2fMax,0, 0.0f) , cv::DMatch(matches2fMax,0, 0.0f) };
                    matches1f_e = std::upper_bound(matches1f.begin(), matches1f.end(), itt,
                        [matches2fMax](std::vector<cv::DMatch> lhs, std::vector<cv::DMatch> rhs) -> bool { return lhs.front().queryIdx < rhs.front().queryIdx; });
                    if (matches1f_e != matches1f.end())
                        matches1f_e++;
                }

                // (Symmetry test) Filter non-symmetrical matches
                // for all matches image 1 -> image 2
                for (std::list<std::vector<cv::DMatch> >::const_iterator matchIterator1 = matches1f_s; matchIterator1 != matches1f_e; ++matchIterator1) {
                    // for all matches image 2 -> image 1
                    for (std::list<std::vector<cv::DMatch> >::const_iterator matchIterator2 = matches2f.begin(); matchIterator2 != matches2f.end(); ++matchIterator2) {

                        // Match symmetry test
                        if (matchIterator1->front().queryIdx == matchIterator2->front().trainIdx &&
                            matchIterator2->front().queryIdx == matchIterator1->front().trainIdx) {

                            // add symmetrical match
                            matches.push_back(cv::DMatch(matchIterator1->front().queryIdx, matchIterator1->front().trainIdx, matchIterator1->front().distance));

                            matches2f.erase(matchIterator2);
                            break; // next match in image 1 -> image 2
                        }
                    }
                }
            }
            else { // old implementation
                // Remove matches for which NN ratio is > than threshold
                // clean image 1 -> image 2 matches
                ratioTest(matches12, minRatio);
                // clean image 2 -> image 1 matches
                ratioTest(matches21, minRatio);

                // Remove non-symmetrical matches
                symmetryTest(matches12, matches21, matches);
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
    cv::Mat& homography,
    const int minNumberMatchesAllowed
    )
{
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
    return matches.size() > inliers.size() * 0.3;
}
