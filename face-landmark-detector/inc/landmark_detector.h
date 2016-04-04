//
// Created by akomandyr on 28.03.16.
//

#ifndef LANDMARK_DETECTOR_H
#define LANDMARK_DETECTOR_H
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

class FaceLandmarkDetector
{
public:
    FaceLandmarkDetector();
    std::vector<dlib::point> DetectFaceLandmarks(const std::string& src);

private:
    dlib::frontal_face_detector detector;

};

#endif //LANDMARK_DETECTOR_H