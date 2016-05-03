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
    ~FaceMorpher();

    int MorphFace(const std::string& src1, const std::string& src2, const std::string& dst);



private:
    std::vector<cv::Vec6f> GetTriangles(cv::Vec6f& firstvector);
    std::vector<cv::Vec6f> GetTriangles(const std::string& src);

    int InsertLastPoints(const cv::Mat& img, std::vector<cv::Point2f>& points);

    void ApplyAffineTransform(cv::Mat& warpImage, cv::Mat& src, std::vector<cv::Point2f>& srcTri, std::vector<cv::Point2f>& dstTri);

    void CalculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &indicesTri);

    std::vector<cv::Point2f> GetAveragePOints(std::vector<cv::Point2f>& src1, std::vector<cv::Point2f>& src2);

    void MorphTriangle(cv::Mat& img1, cv::Mat& img2, cv::Mat& img, std::vector<cv::Point2f>& t1, std::vector<cv::Point2f>& t2, std::vector<cv::Point2f>& t, double alpha);

    FaceLandmarkDetectorPtr m_landmarkDetector;
};
typedef std::shared_ptr<FaceMorpher> FaceMorpherPtr;
#endif //FACEMORPHER_FACEMORPHER_H


