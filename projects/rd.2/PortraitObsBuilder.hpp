#ifndef PORTRAITOBSBUILDER_HPP
#define PORTRAITOBSBUILDER_HPP

#include <vector>

#include <PortraitObs.hpp>
#include <ARPipeline.hpp>


class PortraitObsBuilder {
public:
    static std::vector<PortraitObs> create(const ARPipeline & );

    static void Test(const std::vector<PortraitObs> &);
};

#endif // PORTRAITOBSBUILDER_HPP