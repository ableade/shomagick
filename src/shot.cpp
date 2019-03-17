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
    orientation = imageExif.orientation;
}

std::tuple<ShoRowVector3d, ShoRowVector3d, ShoRowVector3d> Shot::getOrientationVectors() const
{
    if (metadata.orientation == 1) {
       
        return std::make_tuple(getPose().getRotationMatrix().row(0), getPose().getRotationMatrix().row(1), getPose().getRotationMatrix().row(2));
    }
    return std::make_tuple(cv::Mat(), cv::Mat(), cv::Mat());
}
