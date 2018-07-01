#include <PortraitObs.hpp>

void 
PortraitObs::Build(PatternTrackingInfo & me, std::vector<PatternTrackingInfo> & other) {

    // Store 3d pattern position in CAM coordinate system
    m_position = cv::Mat(1, 3, CV_32FC1, me.pose3d.t().data).clone();

    // Found inversion matrix for me
    Matrix44            me44 = me.pose3d.getMat44();
    const cv::Mat       M1 = cv::Mat(4, 4, CV_32FC1, me44.data);
    const cv::Mat       M1inv = M1.inv();

    // Found transform matrices from pattern CS to other patterns
    for (size_t i = 0; i < other.size(); i++) {
        Matrix44        other44 = other[i].pose3d.getMat44();
        const cv::Mat   M2 = cv::Mat(4, 4, CV_32FC1, other44.data);
        const cv::Mat   K12 = M2 * M1inv;

        if (false) // for debug and prove the matrix multiplying order
        {
            cv::Mat             zeroPnt = cv::Mat::zeros(1, 4, CV_32F); zeroPnt.at<float>(0, 3) = 1;
            cv::Mat             expectedPose3d = K12 * M1;
            //
            // These points should be equal
            cv::Mat             realC = zeroPnt * M2;
            cv::Mat             expectedC = zeroPnt * expectedPose3d;
            //
            std::cout << realC << expectedC << std::endl;
        }

        m_K12.push_back(K12);
    }
}