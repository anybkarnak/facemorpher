//
// Created by akomandyr on 28.03.16.
//

#include "face_morpher.h"

FaceMorpher::FaceMorpher()
{
    m_landmarkDetector = new FaceLandmarkDetector("shape_predictor_68_face_landmarks.dat");
}

int FaceMorpher::MorphFace(const std::string& src1, const std::string& src2, const std::string& dst) {

    std::vector<dlib::point> src1_points = m_landmarkDetector->DetectFaceLandmarks(src1);
    return -1;
}

int FaceMorpher::InsertLastPoints(const cv::Mat& img, std::vector<cv::Point2f>& points)
{///insert last points
    points.push_back(cv::Point2f(1, 1));
    points.push_back(cv::Point2f(img.cols - 1, img.rows - 1));

    points.push_back(cv::Point2f(1, img.rows - 1));
    points.push_back(cv::Point2f(img.cols - 1, 1));

    points.push_back(cv::Point2f(1, img.rows / 2));
    points.push_back(cv::Point2f(img.cols / 2, 1));

    points.push_back(cv::Point2f(img.cols / 2, img.rows - 1));
    points.push_back(cv::Point2f(img.cols - 1, img.rows / 2));
}