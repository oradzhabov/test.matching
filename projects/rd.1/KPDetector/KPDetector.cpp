#include "KPDetector.h"

bool FindCameraMatrices(const cv::Mat& K,
    const cv::Mat& Kinv,
    const cv::Mat& distcoeff,
    const std::vector<cv::KeyPoint>& imgpts1,
    const std::vector<cv::KeyPoint>& imgpts2,
    std::vector<cv::KeyPoint>& imgpts1_good,
    std::vector<cv::KeyPoint>& imgpts2_good,
    cv::Matx34d& P,
    cv::Matx34d& P1,
    std::vector<cv::DMatch>& matches,
    std::vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
    , const cv::Mat& img_1,
    const cv::Mat& img_2
#endif
);

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
    std::vector<cv::DMatch>         good_matches;
    std::vector<cv::KeyPoint>		good_keypoints;
    std::vector<cv::KeyPoint>		good_rhs_keypoints;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < 0.6 * matches[i][1].distance && matches[i][0].distance <= 300) {
            good_matches.push_back(matches[i][0]);
            good_keypoints.push_back(m_keypoints[matches[i][0].trainIdx]);
            good_rhs_keypoints.push_back(rhs.m_keypoints[matches[i][0].queryIdx]);
        }
    }

    {
        cv::Mat img_matches;
        cv::drawMatches(rhs.m_img, rhs.m_keypoints, m_img, m_keypoints,
            good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("img_matches", img_matches);
        //cv::waitKey(1);
    }

    if (good_matches.size() < 3) {
        cv::waitKey(1000);
        return -1;
    }

    // Camera intristic parameter matrix
    // I did not calibration
    cv::Mat K = (cv::Mat_<double>(3, 3) <<   500, 0, m_img.cols / 2,
                                            0, 500, m_img.rows / 2,
                                            0, 0, 1);
    //cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1); K.diag() = 1; ///< fictive calibration matrix
    cv::Matx34d P(  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0),
                P1( 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);
    std::vector<CloudPoint>         outCloud;
    const cv::Mat                   distcoeff;              ///< nothing to do
    std::vector<cv::KeyPoint>		goodF_keypoints;        ///< Keypoints which passed fundamental matrix forming test
    std::vector<cv::KeyPoint>		goodF_rhs_keypoints;    ///< Keypoints which passed fundamental matrix forming test

    bool success = FindCameraMatrices(K, K.inv(), distcoeff,
                                        good_keypoints, good_rhs_keypoints,
                                        goodF_keypoints, goodF_rhs_keypoints,
                                        P, P1,
                                        std::vector<cv::DMatch>(),//good_matches,
                                        outCloud
#ifdef __SFM__DEBUG__
                                        , m_img,
                                        rhs.m_img
#endif
    );



    printf("-- Matches Nb: %d \n", matches.size());
    printf("-- Good Matches Nb: %d \n", good_matches.size());
    printf("-- Matches Passed Fundmaental Matrix form test Nb: %d \n", goodF_keypoints.size());

    if (!success) {
        //cv::waitKey(1000);
        //return -1;
    }

    //
    for (size_t i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        rhs.m_matched_pts.push_back(rhs.m_keypoints[good_matches[i].queryIdx].pt);
        m_matched_pts.push_back(m_keypoints[good_matches[i].trainIdx].pt);
    }

    return 0;
}