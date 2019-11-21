#pragma once
#include <opencv2/features2d.hpp>

extern "C" {
#include "vl/covdet.h"
#include "vl/sift.h"
}
const float HAHOG_PEAK_THRESHOLD = 0.00001;
const int HAHOG_EDGE_TRESHOLD = 10;
const bool HAHOG_NORMALIZE_TO_UCHAR = false;

inline VlCovDet * initializeCovdet() {
    VlCovDet * covdet = vl_covdet_new(VL_COVDET_METHOD_HESSIAN);
    return covdet;
}

inline void freeCovdet(VlCovDet* covdet) {
    vl_covdet_delete(covdet);
}


struct CovariantDetectorDeleter
{
    void operator()(VlCovDet* cov) const
    {
        freeCovdet(cov);
    }
};

using unique_vl_COVDET = std::unique_ptr<VlCovDet, CovariantDetectorDeleter>;

static_assert(
    sizeof(VlCovDet*) == sizeof(unique_vl_COVDET),
    "unique covariant pointer has unexpected size"
    );

class HahogFeatureDetector : public cv::Feature2D {
    
private:
    float peakTreshhold_;
    int edgeThreshold_;
    int featuresSize_;
    bool useAdaptiveSupression_;
    VlSiftFilt* sift_;

public:
    virtual ~HahogFeatureDetector();

    HahogFeatureDetector(int targetNumFeatures, float peakThreshold, int edgeThreshold,  bool useAdaptiveSupression);

    HahogFeatureDetector();

    static cv::Ptr<HahogFeatureDetector> create(int target_num_features = 8000, float peakThreshold = HAHOG_PEAK_THRESHOLD,
        int edgeThreshold = HAHOG_EDGE_TRESHOLD,
        bool use_adaptive_suppression = false);

    void detect(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray()
    ) override;

    void compute(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors
    ) override;

    void detectAndCompute(
        cv::InputArray image,
        cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray& descriptors,
        bool useProvidedKeypoints = false
    ) override;

};