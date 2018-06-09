#include "KPDetector.h"

KPDetector::KPDetector() {
    m_detector = cv::BRISK::create();
}

const int KPDetector::detect(const cv::Mat & img) {

    m_keypoints.clear();
    m_descriptors.release();

    m_img = img.clone();
    m_detector->detectAndCompute(m_img, cv::Mat(), m_keypoints, m_descriptors);

    return 0;
}

int KPDetector::match(KPDetector & rhs) {

    m_matched_pts.clear();
    rhs.m_matched_pts.clear();

    cv::BFMatcher                           matcher;
    std::vector<std::vector<cv::DMatch>>	matches;

    // Seach part of RHS in this object, because RHS(new image) contin OLD part and NEW part. OLD part represented by THIS object
    matcher.knnMatch(rhs.m_descriptors, m_descriptors, matches, 2);

    //-- Find only "good" matches. Filter by aspect between 2 neighbours and max distance
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.6 * matches[i][1].distance && matches[i][0].distance <= 300)
            good_matches.push_back(matches[i][0]);
    }


    printf("-- Matches Nb: %d \n", matches.size());
    printf("-- Good Matches Nb: %d \n", good_matches.size());

    {
        cv::Mat img_matches;
        cv::drawMatches(rhs.m_img, rhs.m_keypoints, m_img, m_keypoints,
            good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("img_matches", img_matches);
    }

    if (good_matches.size() == 0) {
        cv::waitKey(1000);
        return -1;
    }

    //
    for (size_t i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        rhs.m_matched_pts.push_back(rhs.m_keypoints[good_matches[i].queryIdx].pt);
        m_matched_pts.push_back(m_keypoints[good_matches[i].trainIdx].pt);
    }

    return 0;
}