#pragma once
#include "flightsession.h"
#include "reconstructor.h"
#include "shotracksgraph.h"

struct RotationOnlyReconstructabilityScore
{
    void operator() (const ShoTracksGraph &tg,
        std::vector<CommonTrack>& commonTracks,
        const FlightSession& flight
        ) {
            auto imageNodes = tg.getImageNodes();
            for (auto &track : commonTracks) {
                cv::Mat mask;
                bool success{};
                cv::Mat essentialMat, rotation, translation;
                std::tie(success, essentialMat, rotation, translation) = twoViewReconstructionRotationOnly(track, mask, tg, flight);
                if (success) {
                    track.rScore = computeReconstructabilityScore(track.commonTracks.size(), mask);
                }
                else {
                    track.rScore = 0.0;
                }
            }

            sort(std::begin(commonTracks), std::end(commonTracks),
                [](const CommonTrack& a, const CommonTrack& b) { return a.rScore > b.rScore; });
    };

    float computeReconstructabilityScore(int tracks, cv::Mat mask,
        int tresh = 3.0) {
        //We use rotation only corrspondence to compute the reconstruction score
        auto inliers = cv::countNonZero(mask);
        auto outliers = tracks - inliers;
        auto ratio = float(inliers) / tracks;
        auto outlierRatio = 1.0 - ratio;
        if (outlierRatio >= 0.25)
            return outliers;
        else
            return 0;
    }
};

struct SnavelyReconstructionabilityScore {

    void operator() (const ShoTracksGraph &tg,
        std::vector<CommonTrack>& commonTracks,
        const FlightSession& flight
        ) {
        auto imageNodes = tg.getImageNodes();
        for (auto &track : commonTracks) {
            cv::Mat mask;
#if 0
            bool success{};
            Mat essentialMat, rotation, translation;
            std::tie(success, essentialMat, rotation, translation) = twoViewReconstructionRotationOnly(track, mask);
            //std::tie(success, essentialMat, rotation, translation) = recoverTwoViewPoseWithHomography(track, mask);
#else
            auto[success, essentrialMat, rotation, translation] = recoverTwoViewPoseWithHomography(
                track, mask, flight, tg);
#endif
            if (success) {
                track.rScore = computeReconstructabilityScore(track.commonTracks.size(), mask);
            }
            else {
                track.rScore = 1.0;
            }
        }

        sort(std::begin(commonTracks), std::end(commonTracks),
            [](const CommonTrack& a, const CommonTrack& b) { return a.rScore < b.rScore; });

    };

    float computeReconstructabilityScore(int tracks, cv::Mat mask,
        int tresh =3.0) {
        auto inliers = cv::countNonZero(mask);
        auto outliers = tracks - inliers;
        auto ratio = float(inliers) / tracks;
        if (tracks > 100)
            return ratio;
        else
            return 1.0;
    }
};

struct MatchesCountReconstructabilityScore {
    void operator() (const ShoTracksGraph &tg,
        std::vector<CommonTrack>& commonTracks,
        const FlightSession& flight
        ) {
        sort(std::begin(commonTracks), std::end(commonTracks),
            [](const CommonTrack& a, const CommonTrack& b) { return a.commonTracks.size() > b.commonTracks.size(); });
    }
};