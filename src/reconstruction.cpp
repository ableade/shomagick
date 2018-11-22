#include "reconstruction.h"

Reconstructor :: Reconstructor (FlightSession flight, TrackGraph tg): flight (flight), tg(tg)  {};

void Reconstructor::computeEssentialMatrix(string image1,  string image2,  Camera camera) {
    auto allPairMatches = this->flight.loadMatches(image1);
    auto pairwiseMatches = allPairMatches[image2];
    auto features2 = this->flight.loadFeatures(image2);
    auto features1 = this->flight.loadFeatures(image1);

    //align pairwise matches
    for(auto i=0 ; i< pairwiseMatches.size(); ++i) {
        
    }
}
