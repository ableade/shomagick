#pragma once
// Isolating on a namespace to prevent collisions with opencv names
#define _USE_OPENCV
#include "MVSInterface.h" 
#include <map>
#include <opencv2/core.hpp>


namespace csfm {

    class OpenMVSExporter {
    public:
        void AddCamera(
            const std::string &camera_id,
            cv::Matx33d K) {
            MVS::Interface::Platform platform;
            platform.name = camera_id;
            MVS::Interface::Platform::Camera camera;
            camera.K = K;
            camera.R = cv::Matx33d::eye();
            camera.C = cv::Point3_<double>(0, 0, 0);
            platform.cameras.push_back(camera);

            platform_ids_[camera_id] = scene_.platforms.size();
            scene_.platforms.push_back(platform);
        }

        void AddShot(
            const std::string &path,
            const std::string &shot_id,
            const std::string &camera_id,
            cv::Matx33d R,
            cv::Point3_<double> C) {

            int platform_id = platform_ids_[camera_id];
            MVS::Interface::Platform &platform = scene_.platforms[platform_id];

            MVS::Interface::Platform::Pose pose;
            pose.R = R;
            pose.C = C;
            int pose_id = platform.poses.size();
            platform.poses.push_back(pose);

            MVS::Interface::Image image;
            image.name = path;
            image.platformID = platform_id;
            image.cameraID = 0;
            image.poseID = pose_id;

            image_ids_[shot_id] = scene_.images.size();
            scene_.images.push_back(image);
        }

        void AddPoint(
            cv::Point3d coordinates,
            std::vector<std::string> shot_ids) {
         

            MVS::Interface::Vertex vertex;
            vertex.X = coordinates;
            for (auto shot: shot_ids) {
                MVS::Interface::Vertex::View view;
                view.imageID = image_ids_[shot];
                view.confidence = 0;
                vertex.views.push_back(view);
            } 
            scene_.vertices.push_back(vertex);
        }

        void Export(std::string filename) {
            MVS::ARCHIVE::SerializeSave(scene_, filename);
        }

    private:
        std::map<std::string, int> platform_ids_;
        std::map<std::string, int> image_ids_;
        MVS::Interface scene_;
    };


}
