#include <PortraitObsBuilder.hpp>


std::vector<PortraitObs> PortraitObsBuilder::create(const ARPipeline & pipeline) {
    std::vector<PortraitObs> result;

    std::vector<PatternTrackingInfo> foundPatterns;

    const size_t maxPatternCount = pipeline.getPatternsCount();

    // Collect found patterns
    for (size_t i = 0; i < maxPatternCount; i++) {
        if (pipeline.isPatternFound(i))
            foundPatterns.push_back(pipeline.getPatternInfo(i));
    }

    // 
    result.resize(foundPatterns.size());
    for (size_t i = 0; i < foundPatterns.size(); i++) {
        std::vector<PatternTrackingInfo> foundPatternsExceptMe = foundPatterns;
        foundPatternsExceptMe.erase(foundPatternsExceptMe.begin() + i);

        result[i].Build(foundPatterns[i], foundPatternsExceptMe);
    }


    return result;
}

void PortraitObsBuilder::Test(const std::vector<PortraitObs> & obsPortraits) {
    double                      minDist = std::numeric_limits<double>::max();
    int                         nearestPtnInd = -1;

    // Find index of portrait which the most nearest to camera
    for (int i = 0; i < obsPortraits.size(); i++) {
        double length = cv::norm(obsPortraits[i].m_position, cv::NORM_L2);
        if (length < minDist) {
            minDist = length;
            nearestPtnInd = i;
        }
    }

    if (nearestPtnInd >= 0) {

        cv::Mat zeroPnt = cv::Mat::zeros(1, 4, CV_32F); zeroPnt.at<float>(0, 3) = 1;
        for (int i = 0; i < obsPortraits[nearestPtnInd].m_K12.size(); ++i) {
            
            // Find position(in nearest portrait coordinate system) of neighbour portrait
            cv::Mat pos = zeroPnt * obsPortraits[nearestPtnInd].m_K12[i];

            if (true) // show relative pos
                std::cout << pos << std::endl;
            //printf("Nearest Distance: %.2f\tFrom nearest to second: %.2f\n", minDist, cv::norm(cv::Mat(1,3,CV_32F, pos.data), cv::NORM_L2));
        }
    }
}