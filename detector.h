#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using std::vector;
using cv::KeyPoint;
using cv::Feature2D;
using cv::Ptr;
using cv::Mat;
using cv::DMatch;
using std::cout;

using namespace cv::xfeatures2d;


template <class DetectorClass>
struct Detector
{
    Ptr<DetectorClass> detector;

    Detector()
    {
        detector =  DetectorClass::create();
    }

    template<class T>
    void operator()(const T& in, const T& mask, vector<KeyPoint>& pts, T& descriptors, bool useProvided=false )
    {
        detector->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct Matcher
{
    KPMatcher matcher;

    template <class T>
    Matcher(const vector<T>& descriptorSet):matcher{}{
        matcher.add(descriptorSet);
        cout << "Now training descriptor set for all images";
        matcher.train();
    }
    template<class T>
    void match(const T& in1, std::vector< std::vector< DMatch > > &matches, int limit)
    {
        matcher.knnMatch(in1, matches, limit, cv::noArray(), false);
    }
};