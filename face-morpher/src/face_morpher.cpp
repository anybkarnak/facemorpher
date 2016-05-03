//
// Created by akomandyr on 28.03.16.
//

#include "face_morpher.h"

using namespace cv;
using namespace std;

FaceMorpher::FaceMorpher()
{
    m_landmarkDetector = std::make_shared<FaceLandmarkDetector>("shape_predictor_68_face_landmarks.dat");
}

int FaceMorpher::MorphFace(const std::string& src1, const std::string& src2, const std::string& dst) {

    // Read first image
    Mat img1 = imread(src1);
    img1.convertTo(img1, CV_32F);
    // Rectangle to be used with Subdiv2D
    Size size = img1.size();
    Rect rect(0, 0, size.width, size.height);
    // Create a vector of points.
    vector<Point2f> points1;
    std::vector<dlib::point> src1_points = m_landmarkDetector->DetectFaceLandmarks(src1);
    for (unsigned long k = 0; k < src1_points.size(); ++k)
    {
        // cout << "pixel position of " <<k<<"  part: x= " << dl_points[k].x()<<" y= " <<dl_points[k].y()<< endl;
        points1.push_back(Point2f(src1_points[k].x(), src1_points[k].y()));
    }

    InsertLastPoints(img1,  points1);

    // Read second image.
    Mat img2 = imread(src2);
    if(img2.rows!=img1.rows||img2.cols!=img1.cols)
    {
        cv::imwrite("tmp"+src1,img1);
        cv::resize(img2,img2,cv::Size(img1.cols,img1.rows));
        cv::imwrite(src1,img1);
    }

    img1.convertTo(img1, CV_32F);

    // Rectangle to be used with Subdiv2D
    Size size1 = img1.size();
    Rect rect1(0, 0, size.width, size.height);

    // Create a vector of points.
    vector<Point2f> points2;

    std::vector<dlib::point> src2_points2 = m_landmarkDetector->DetectFaceLandmarks(src2);

    for (unsigned long k = 0; k < src2_points2.size(); ++k)
    {
        //cout << "pixel position of " <<k<<"  part: x= " << dl_points1[k].x()<<" y= " <<dl_points1[k].y()<< endl;
        points2.push_back(Point2f(src2_points2[k].x(), src2_points2[k].y()));
    }

    InsertLastPoints(img2,  points2);

    std::vector<Point2f> avgPoints = GetAveragePOints(points1, points2); //the same order


    Mat img1Warped = img1.clone();

    vector< vector<int> > dt;
    Rect rectWarped(0, 0, img1Warped.cols, img1Warped.rows);
    CalculateDelaunayTriangles(rectWarped, avgPoints, dt);

    // imshow("dl1", img1/255);
    vector<Point2f> morphPoints;
    double alpha = 0.5;//HARCODE WARNING
    //empty average image
    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);
    //compute weighted average point coordinates
    for (int i = 0; i < points1.size(); i++)
    {
        float x, y;
        x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
        y = (1 - alpha) * points1[i].y + alpha * points2[i].y;

        morphPoints.push_back(Point2f(x, y));
    }


    for (int i = 0; i < dt.size(); i ++)
    {
        // Triangles
        vector<Point2f> t1, t2, t;


        for(size_t j = 0; j < 3; j++)
        {
            t1.push_back(points1[dt[i][j]]);
            t2.push_back(points2[dt[i][j]]);
            t.push_back(morphPoints[dt[i][j]]);
        }

        MorphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);
    }

    imwrite(dst, imgMorph );
    //imwrite(path1+"Morphed_Face.jpg", imgMorph );
    return 0;
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

void FaceMorpher::ApplyAffineTransform(cv::Mat& warpImage, cv::Mat& src, std::vector<cv::Point2f>& srcTri, std::vector<cv::Point2f>& dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

void FaceMorpher::CalculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &indicesTri)
{
// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5 ]);

        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] = k;

            indicesTri.push_back(ind);
        }
    }
}


void FaceMorpher::MorphTriangle(cv::Mat& img1, cv::Mat& img2, cv::Mat& img, std::vector<cv::Point2f>& t1, std::vector<cv::Point2f>& t2, std::vector<cv::Point2f>& t, double alpha)
{
// Find bounding rectangle for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect, tRect;
    vector<Point> tRectInt;
    for (int i = 0; i < 3; i++)
    {
        tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
        tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y)); // for fillConvexPoly

        t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
        t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
    fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);

    Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
    Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());

    ApplyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    ApplyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

    // Alpha blend rectangular patches
    Mat imgRect00 = (1.0 - alpha) * warpImage1;
    Mat imgRect01 = alpha * warpImage2;
    imgRect00.convertTo(imgRect00, CV_32FC3);
    imgRect01.convertTo(imgRect01, CV_32FC3);

    Mat imgRect = imgRect00 + imgRect01;

    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect, mask, imgRect);
    multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
    img(r) = img(r) + imgRect;

}


std::vector<cv::Point2f> FaceMorpher::GetAveragePOints(std::vector<cv::Point2f>& src1, std::vector<cv::Point2f>& src2)
{
    std::vector<cv::Point2f> avgPoints;
    for (int i = 0; i < src1.size(); i++)
    {
        float x, y;
        x =  (src1[i].x +  src2[i].x)*0.5;
        y =  (src1[i].y +  src2[i].y)*0.5;

        avgPoints.push_back(Point2f(x, y));
    }

    return avgPoints;
}

FaceMorpher::~FaceMorpher()
{

}