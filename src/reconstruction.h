#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "flightsession.h"
#include "shotracking.h"
#include "camera.h"

class Reconstructor {
    private:
        FlightSession flight;
        TrackGraph tg;

    public:
        Reconstructor(FlightSession flight, TrackGraph tg);
        Reconstructor(FlightSession flight, ShoTracker tracker);
        void computeEssentialMatrix(string image1,  string image2, Camera camera);

};

#endif