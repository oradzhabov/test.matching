#include <PortraitObs.hpp>

PortraitObs::PortraitObs(PatternTrackingInfo & me, std::vector<PatternTrackingInfo> & other) {

    // Store 3d pattern position in CAM coordinate system
    m_position = cv::Mat(1, 3, CV_32FC1, me.pose3d.t().data).clone();

    Matrix44        meInv44 = me.pose3dInv.getMat44();
    const cv::Mat   M1inv = cv::Mat(4, 4, CV_32FC1, meInv44.data);

    // Found transform matrices from pattern CS to other patterns
    for (size_t i = 0; i < other.size(); i++) {
        Matrix44        other44 = other[i].pose3d.getMat44();
        const cv::Mat   M2 = cv::Mat(4, 4, CV_32FC1, other44.data);
        const cv::Mat   K12 = M1inv * M2;

        m_K12.push_back(K12);
    }
}