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

cv::Matx34d P(1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0);

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
        if (matches[i][0].distance < 0.6 * matches[i][1].distance && matches[i][0].distance <= 100) {
		//if (matches[i][0].distance < 0.6 * matches[i][1].distance) {
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
    cv::Mat K = (cv::Mat_<double>(3, 3) <<   400, 0, m_img.cols / 2,
                                            0, 400, m_img.rows / 2,
                                            0, 0, 1);
    //cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1); K.diag() = 1; ///< fictive calibration matrix
	cv::Matx34d P1( 1, 0, 0, 0,
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

	if (success && false) {
		cv::Matx44d P_4x4(P(0, 0), P(0, 1), P(0, 2), P(0, 3),
			P(1, 0), P(1, 1), P(1, 2), P(1, 3),
			P(2, 0), P(2, 1), P(2, 2), P(2, 3),
			0, 0, 0, 1);
		cv::Matx44d P1_4x4(P1(0, 0), P1(0, 1), P1(0, 2), P1(0, 3),
			P1(1, 0), P1(1, 1), P1(1, 2), P1(1, 3),
			P1(2, 0), P1(2, 1), P1(2, 2), P1(2, 3),
			0, 0, 0, 1);
		P_4x4 = P_4x4 * P1_4x4;
		P(0, 0) = P_4x4(0, 0);P(0, 1) = P_4x4(0, 1);P(0, 2) = P_4x4(0, 2);P(0, 3) = P_4x4(0, 3);
		P(1, 0) = P_4x4(1, 0);P(1, 1) = P_4x4(1, 1);P(1, 2) = P_4x4(1, 2);P(1, 3) = P_4x4(1, 3);
		P(2, 0) = P_4x4(2, 0);P(2, 1) = P_4x4(2, 1);P(2, 2) = P_4x4(2, 2);P(2, 3) = P_4x4(2, 3);
		std::cout << "----------------------------------------------------------------------" << std::endl;
		std::cout << P << std::endl;
		//
		// set up points on a plane
		cv::Mat t = rhs.m_img.clone();
		float x = t.cols / 2;
		float y = t.rows / 2;
		float z = 2000;
		std::vector<cv::Vec4d> p3d{ { 0, 0, z, 1 }, { x, 0, z, 1 }, { 0, y, z, 1 }, { x, y, z, 1} };
		for (int i = 0; i < p3d.size(); ++i) {
			cv::Vec4d proj = P_4x4 * p3d[i];
			std::cout << proj << std::endl;
			cv::circle(t, cv::Point(proj[0], proj[1]), 10, cv::Scalar(255,255, 0), -1);
		}
		std::cout << "----------------------------------------------------------------------" << std::endl;
		cv::imshow("t",t);

	}
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