//
// Created by akomandyr on 28.03.16.
//

#include "landmark_detector.h"

FaceLandmarkDetector::FaceLandmarkDetector(const std::string &shapePredictor)
{
    m_detector = dlib::get_frontal_face_detector();
    dlib::deserialize(shapePredictor.c_str()) >> m_shape_predictor;
};

std::vector<dlib::point> FaceLandmarkDetector::DetectFaceLandmarks(const std::string &src)
{
    dlib::array2d<dlib::rgb_pixel> img;
    load_image(img, src.c_str());
    // Make the image larger so we can detect small faces.
    //pyramid_up(img);

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    std::vector<dlib::rectangle> dets = m_detector(img);
    //std::cout << "Number of faces detected: " << dets.size() << std::endl;

    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.

    dlib::full_object_detection shape = m_shape_predictor(img, dets[0]);
    std::vector<dlib::point> points;

    for (unsigned long k = 0; k < shape.num_parts(); ++k)
    {
        points.push_back(shape.part(k));
    }

    return points;
}

FaceLandmarkDetector::~FaceLandmarkDetector()
{

}