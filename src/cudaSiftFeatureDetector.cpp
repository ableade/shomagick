#include "cudaSiftFeatureDetector.h"
#include "cudaImage.h"
#include "cudaSift.h"

using std::vector;
using cv::Mat;
using cv::KeyPoint;

void CudaSiftFeatureDetector::_downloadKeypoints(const SiftData & sd, std::vector<cv::KeyPoint>& keypoints)
{
    auto ptr =  _getSiftPointItr(sd);
    //Point2f _pt, float _size, float _angle = -1, float _response = 0, int _octave = 0, int _class_id = -1)
    for (auto i = 0; i < sd.numPts; ++i) {
        ptr+=i;
        KeyPoint kp{ {ptr->xpos, ptr->ypos}, ptr->scale, ptr->orientation };
        keypoints.push_back(kp);
    }
}

SiftPoint* CudaSiftFeatureDetector::_getSiftPointItr(const SiftData & sd)
{
    SiftPoint *siftItr = nullptr;
#ifdef MANAGEDMEM
    siftItr = sd.m_data;
#else
    if (sd.d_data != nullptr)
        siftItr = sd.d_data;
#endif

    return siftItr;
}

void CudaSiftFeatureDetector::_downloadDescriptors(const SiftData & sd, cv::Mat & descriptors)
{
    auto ptr = _getSiftPointItr(sd);
    //Point2f _pt, float _size, float _angle = -1, float _response = 0, int _octave = 0, int _class_id = -1)
    for (auto i = 0; i < sd.numPts; ++i) {
        ptr += i;
        Mat desc(1, SIFT_BIN_SIZE, CV_32F, ptr->data);
        descriptors.push_back(desc);
    }
}

cv::Ptr<CudaSiftFeatureDetector> CudaSiftFeatureDetector::create() {
    return cv::makePtr<CudaSiftFeatureDetector>();
}

void CudaSiftFeatureDetector::detect(Mat image, vector<KeyPoint>& keypoints, Mat &mask) {
    SiftData siftData;
    InitSiftData(siftData, SIFT_DATA_RESERVE_SIZE, true, true);

    CudaImage img;
    img.Allocate(image.size().width, image.size().height, image.size().width, false, NULL, (float*)image.data);
    img.Download();


    int numOctaves = 5;    /* Number of octaves in Gaussian pyramid */
    float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
    float thresh = 3.5f;   /* Threshold on difference of Gaussians for feature pruning */
    float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
    bool upScale = false;  /* Whether to upscale image before extraction */
    /* Extract SIFT features */
    ExtractSift(siftData, img, numOctaves, initBlur, SIFT_EDGE_THRESHOLD, minScale, upScale);
    _downloadKeypoints(siftData, keypoints);
}

void CudaSiftFeatureDetector::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    auto im = image.getMat().clone();
    auto maskMat = mask.getMat();
    
    SiftData siftData;
    InitSiftData(siftData, SIFT_DATA_RESERVE_SIZE, true, true);

    CudaImage img;
    img.Allocate(image.size().width, image.size().height, image.size().width, false, NULL, (float*)im.data);
    img.Download();


    int numOctaves = 5;    /* Number of octaves in Gaussian pyramid */
    float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
    float thresh = 3.5f;   /* Threshold on difference of Gaussians for feature pruning */
    float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
    bool upScale = false;  /* Whether to upscale image before extraction */
    /* Extract SIFT features */
    ExtractSift(siftData, img, numOctaves, initBlur, SIFT_EDGE_THRESHOLD, minScale, upScale);
    _downloadKeypoints(siftData, keypoints);
}

void CudaSiftFeatureDetector::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray & descriptors, bool useProvidedKeypoints)
{
    if (!useProvidedKeypoints) {
        detectAndCompute(image.getMat(), mask.getMat(), keypoints, descriptors);
        return;
    }
    //TODO use provided keypoints
}

void CudaSiftFeatureDetector::detectAndCompute(Mat image, cv::Mat & mask, vector<cv::KeyPoint>& keypoints, Mat & descriptors)
{
    SiftData siftData;
    InitSiftData(siftData, SIFT_DATA_RESERVE_SIZE, true, true);

    CudaImage img;
    img.Allocate(image.size().width, image.size().height, image.size().width, false, NULL, (float*)image.data);
    img.Download();


    int numOctaves = 5;    /* Number of octaves in Gaussian pyramid */
    float initBlur = 1.0f; /* Amount of initial Gaussian blurring in standard deviations */
    float thresh = 3.5f;   /* Threshold on difference of Gaussians for feature pruning */
    float minScale = 0.0f; /* Minimum acceptable scale to remove fine-scale features */
    bool upScale = false;  /* Whether to upscale image before extraction */
    /* Extract SIFT features */
    ExtractSift(siftData, img, numOctaves, initBlur, SIFT_EDGE_THRESHOLD, minScale, upScale);
    _downloadKeypoints(siftData, keypoints);
    _downloadDescriptors(siftData, descriptors);
}




