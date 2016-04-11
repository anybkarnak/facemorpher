//
// Created by akomandyr on 28.03.16.
//

#ifndef FACEMORPHER_H
#define FACEMORPHER_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "landmark_detector.h"

#include <string>

class FaceMorpher
{
public:
    FaceMorpher();
    int MorphFace(const std::string& src1, const std::string& src2, const std::string& dst);
    int InsertLastPoints(const cv::Mat& img, std::vector<cv::Point2f>& points);
private:
    std::vector<cv::Vec6f> GetTriangles(cv::Vec6f& firstvector);
    std::vector<cv::Vec6f> GetTriangles(const std::string& src);
    FaceLandmarkDetector* m_landmarkDetector;

};
#endif //FACEMORPHER_FACEMORPHER_H
