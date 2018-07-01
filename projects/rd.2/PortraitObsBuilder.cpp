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