#include "HahogFeatureDetector.h"
#include "bootstrap.h"
#include <iostream>

extern "C" {
#include "vl/covdet.h"
#include "vl/sift.h"
#include <time.h>
}

using cv::KeyPoint;

HahogFeatureDetector::~HahogFeatureDetector()
{
    vl_sift_delete(sift_);
    vl_covdet_delete(covdet_);
}

HahogFeatureDetector::HahogFeatureDetector(int targetNumFeatures, float peakThreshold, int edgeThreshold,  bool useAdaptiveSupression) :featuresSize_(targetNumFeatures), peakTreshhold_(peakThreshold),
edgeThreshold_(edgeThreshold),  useAdaptiveSupression_(useAdaptiveSupression)
{
    sift_ = vl_sift_new(16, 16, 1, 3, 0);
    covdet_ = vl_covdet_new(VL_COVDET_METHOD_HESSIAN);
}

HahogFeatureDetector::HahogFeatureDetector()
{
}

cv::Ptr<HahogFeatureDetector> HahogFeatureDetector::create(int targetNumFeatures, float peakThreshold, int edgeThreshold,  bool useAdaptiveSupression)
{
    return cv::makePtr<HahogFeatureDetector>(targetNumFeatures, peakThreshold, edgeThreshold,  useAdaptiveSupression);
}

void HahogFeatureDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    auto imFlat = image.getMat().reshape(1, 1);
    // set various parameters (optional)
    vl_covdet_set_first_octave(covdet_, 0);
    //vl_covdet_set_octave_resolution(covdet, octaveResolution);
    vl_covdet_set_peak_threshold(covdet_, peakTreshhold_);
    vl_covdet_set_edge_threshold(covdet_, edgeThreshold_);
    
    vl_covdet_set_target_num_features(covdet_, featuresSize_);
    vl_covdet_set_use_adaptive_suppression(covdet_, useAdaptiveSupression_);

    vl_covdet_put_image(covdet_, reinterpret_cast<float*>(imFlat.data), image.cols(), image.rows());

    //clock_t t_scalespace = clock();

    vl_covdet_detect(covdet_);

    //clock_t t_detect = clock();

    // compute the affine shape of the features (optional)
    //vl_covdet_extract_affine_shape(covdet);

    //clock_t t_affine = clock();

    // compute the orientation of the features (optional)
    vl_covdet_extract_orientations(covdet_);

    // get feature descriptors
    vl_size numFeatures = vl_covdet_get_num_features(covdet_);
    VlCovDetFeature const *feature = (VlCovDetFeature const *)vl_covdet_get_features(covdet_);
    vl_index i;

    vl_sift_set_magnif(sift_, 3.0);
    for (i = 0; i < (signed)numFeatures; ++i) {
        KeyPoint kp;
        const VlFrameOrientedEllipse &frame = feature[i].frame;
        float det = frame.a11 * frame.a22 - frame.a12 * frame.a21;
        float size = sqrt(fabs(det));
        float angle = atan2(frame.a21, frame.a11) * 180.0f / M_PI;
        kp.pt.x = frame.x;
        kp.pt.y = frame.y;
        kp.size = size;
        kp.angle = angle;

        keypoints.push_back(kp);
    }
}

void HahogFeatureDetector::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
}

void HahogFeatureDetector::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    cv::Mat mat = image.getMat();
    cv::Mat imFlat = mat.reshape(1, 1).clone();
    if (imFlat.isContinuous()) {
        std::cout << "We have continious \n";
    }
    std::cout << "Rows * cols m flat " << imFlat.rows * imFlat.cols << "\n";
    std::cout << "Rows *m flat " << imFlat.rows << "\n";
    std::cout << "Cols *m flat " << imFlat.cols << "\n";
    // set various parameters (optional)
    vl_covdet_set_first_octave(covdet_, 0);
    //vl_covdet_set_octave_resolution(covdet, octaveResolution);
    vl_covdet_set_peak_threshold(covdet_, peakTreshhold_);
    vl_covdet_set_edge_threshold(covdet_, edgeThreshold_);

    vl_covdet_set_target_num_features(covdet_, featuresSize_);
    vl_covdet_set_use_adaptive_suppression(covdet_, useAdaptiveSupression_);

    std::vector<float> buffer;
    buffer.assign((float*)imFlat.data, (float*)imFlat.data + imFlat.total());
    //const float* buffer = imFlat.ptr<float>(0);
    //vl_covdet_put_image(covdet_, buffer, image.cols(), image.rows());

    vl_covdet_put_image(covdet_, buffer.data(), image.cols(), image.rows());

    //clock_t t_scalespace = clock();

    vl_covdet_detect(covdet_);

    //clock_t t_detect = clock();

    // compute the affine shape of the features (optional)
    //vl_covdet_extract_affine_shape(covdet);

    //clock_t t_affine = clock();

    // compute the orientation of the features (optional)
    vl_covdet_extract_orientations(covdet_);

    // get feature descriptors
    vl_size numFeatures = vl_covdet_get_num_features(covdet_);
    VlCovDetFeature const *feature = (VlCovDetFeature const *)vl_covdet_get_features(covdet_);
    vl_index i;
    vl_size dimension = 128;
    vl_index patchResolution = 15;
    double patchRelativeExtent = 7.5;
    double patchRelativeSmoothing = 1;
    vl_size patchSide = 2 * patchResolution + 1;
    double patchStep = static_cast<double>(patchRelativeExtent) / patchResolution;
    std::vector<float> desc(dimension * numFeatures);
    std::vector<float> patch(patchSide * patchSide);
    std::vector<float> patchXY(2 * patchSide * patchSide);

    vl_sift_set_magnif(sift_, 3.0);
    for (i = 0; i < (signed)numFeatures; ++i) {
        KeyPoint kp;
        const VlFrameOrientedEllipse &frame = feature[i].frame;
        float det = frame.a11 * frame.a22 - frame.a12 * frame.a21;
        float size = sqrt(fabs(det));
        float angle = atan2(frame.a21, frame.a11) * 180.0f / M_PI;
        kp.pt.x = frame.x;
        kp.pt.y = frame.y;
        kp.size = size;
        kp.angle = angle;

        keypoints.push_back(kp);

        vl_covdet_extract_patch_for_frame(covdet_,
            &patch[0],
            patchResolution,
            patchRelativeExtent,
            patchRelativeSmoothing,
            frame);

        vl_imgradient_polar_f(&patchXY[0], &patchXY[1],
            2, 2 * patchSide,
            &patch[0], patchSide, patchSide, patchSide);

        vl_sift_calc_raw_descriptor(sift_,
            &patchXY[0],
            &desc[dimension * i],
            (int)patchSide, (int)patchSide,
            (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2,
            (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
            VL_PI / 2);
    }
    cv::Mat descriptorsMat(static_cast<int>(numFeatures), static_cast<int>(dimension), CV_32FC1, desc.data());
    descriptors.assign(descriptorsMat);
}
