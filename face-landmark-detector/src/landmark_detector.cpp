//
// Created by akomandyr on 28.03.16.
//

#include "landmark_detector.h"

FaceLandmarkDetector::FaceLandmarkDetector()
{
    detector = dlib::get_frontal_face_detector();
};