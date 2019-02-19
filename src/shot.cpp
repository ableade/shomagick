#include "shot.h"
#include "flightsession.h"

ShotMetadata::ShotMetadata() :gpsPosition(), gpsDop(), captureTime(), orientation() {}

ShotMetadata::ShotMetadata(
    cv::Point3d gpsPosition, 
    double dop, 
    double captureTime, 
    int orientation
): gpsPosition(gpsPosition), gpsDop(dop), captureTime(captureTime),
orientation(orientation){}

ShotMetadata::ShotMetadata(ImageMetadata imageExif, const FlightSession & flight)
{
    gpsPosition = imageExif.location.getTopcentricLocationCoordinates(flight.getReferenceLLA());
    gpsDop = imageExif.location.dop;
    captureTime = imageExif.captureTime;
}

cv::Mat Shot::getOrientationVectors()
{
    return cv::Mat();
}
