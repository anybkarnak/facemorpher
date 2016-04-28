#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "movie_creator.h"
#include "landmark_detector.h"


using namespace std;
using namespace cv;

std::vector<cv::Vec6f> GetTriangles(std::vector<cv::Vec6f>& firstvector, std::vector<cv::Vec6f>& secondvector);

// Draw a single point
static void draw_point(Mat& img, Point2f fp, Scalar color, std::string text)
{
    putText(img, text, fp,
            FONT_HERSHEY_COMPLEX_SMALL, 0.3, color, 1, CV_AA);
    //circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
}

// Draw delaunay triangles
static void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{

    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}

//get delaunau triangles
static vector<Point2f> getVectorTriangles(Mat& img, Subdiv2D& subdiv, vector<Vec6f> firstvector  = vector<Vec6f>())
{
    vector<Point2f> points;
    vector<Vec6f> triangleList;

    if (firstvector.size() == 0)
    {
        subdiv.getTriangleList(triangleList);
    }
    else
    {
        vector<Vec6f> tmptrianglelist;
        subdiv.getTriangleList(tmptrianglelist);
        triangleList = GetTriangles(firstvector, tmptrianglelist);
    }

    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            points.push_back(pt[0]);
            points.push_back(pt[1]);
            points.push_back(pt[2]);
        }
    }

    std::cout<<" triangles = "<<triangleList.size()<<std::endl;

    return points;
}


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat& warpImage, Mat& src, vector<Point2f>& srcTri, vector<Point2f>& dstTri)
{

    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform just found to the src image
    warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriangle(Mat& img1, Mat& img2, Mat& img, vector<Point2f>& t1, vector<Point2f>& t2, vector<Point2f>& t, double alpha)
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

    applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);

    // Alpha blend rectangular patches
    Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;

    // Copy triangular region of the rectangular patch to the output image
    multiply(imgRect, mask, imgRect);
    multiply(img(r), Scalar(1.0, 1.0, 1.0) - mask, img(r));
    img(r) = img(r) + imgRect;


}

std::vector<cv::Vec6f> GetTriangles(std::vector<cv::Vec6f>& firstvector, std::vector<cv::Vec6f>& secondvector)
{
    std::vector<cv::Vec6f> resultvector;
    int maxR = 500;
    int tri=0;
    std::vector<int> lossTriangles;
    for ( auto it:firstvector)
    {

        bool isAdded=false;
        for(int i=50;i<=maxR; i++)
        {
            auto tmpvec = find_if(secondvector.begin(), secondvector.end(), [ it, i ](const cv::Vec6f& second)
            {


                return (sqrt(abs((pow((second[0] - it[0]), 2) + pow((second[1] - it[1]), 2)))) < i &&
                        sqrt(abs((pow((second[2] - it[2]), 2) + pow((second[3] - it[3]), 2)))) < i &&
                        sqrt(abs((pow((second[4] - it[4]), 2) + pow((second[5] - it[5]), 2)))) < i);
            });

            if (tmpvec == secondvector.end())
            {

                continue;
            }
            else
            {
                resultvector.push_back(*tmpvec);
                secondvector.erase(tmpvec);
                isAdded=true;
                break;
                //resultvector.push_back(it);
            }
        }
        if(!isAdded)
        {
            lossTriangles.push_back(tri);
            resultvector.push_back(it);
        }
        tri++;
    }

    if (resultvector.size() < firstvector.size())
    {
        std::cout << " size<" << std::endl;
    }
    std::cout << " size " <<resultvector.size()<< std::endl;
    std::cout << " lost triangles " <<std::endl;

    for(auto& angle:lossTriangles)
    {
        cv::Vec6f it = firstvector[angle];
        bool isAdded=false;
        for(int i=maxR;i<=maxR*3; i+=5)
        {
            auto tmpvec = find_if(secondvector.begin(), secondvector.end(), [ it, i ](const cv::Vec6f& second)
            {


                return (sqrt(abs((pow((second[0] - it[0]), 2) + pow((second[1] - it[1]), 2)))) < i &&
                        sqrt(abs((pow((second[2] - it[2]), 2) + pow((second[3] - it[3]), 2)))) < i &&
                        sqrt(abs((pow((second[4] - it[4]), 2) + pow((second[5] - it[5]), 2)))) < i);
            });

            if (tmpvec == secondvector.end())
            {

                continue;
            }
            else
            {
                std::cout << angle << std::endl;
                resultvector[angle]=*tmpvec;
                secondvector.erase(tmpvec);

                break;
                //resultvector.push_back(it);
            }
        }


        std::cout << angle <<"++"<< std::endl;
    }


    return resultvector;
}

int InsertLastPoints(const cv::Mat& img, std::vector<cv::Point2f>& points)
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

// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri){

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

            delaunayTri.push_back(ind);
        }
    }

}

int main(int argc,      // Number of strings in array argv
         char* argv[])  // Array of command-line argument strings
{
    int count;
// Display each command-line argument.
    cout << "\nCommand-line arguments:\n";
    for (count = 0; count < argc; count++)
        cout << "  argv[" << count << "]   "
        << argv[count] << "\n";
    cout << "Hello, World!" << endl;

    std::string path;
    std::string path1;

    if(argc == 3)
    {
         path = argv[1];
         path1 = argv[2];
    }
    else
    {
         path = "ant.jpg";
         path1 = "nast.jpg";
    }

    FaceLandmarkDetector* fld = new FaceLandmarkDetector("shape_predictor_68_face_landmarks.dat");

    // Define window names
    string win_delaunay = "Delaunay Triangulation";
    // Turn on animation while drawing triangles

    bool animate = false;
    Scalar delaunay_color(0, 255, 255), points_color(0, 0, 255);

    // Define colors for drawing.

    // Read in the image.
    Mat img = imread(path);

    img.convertTo(img, CV_32F);

    // Rectangle to be used with Subdiv2D
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    // Create a vector of points.
    vector<Point2f> points;

    std::vector<dlib::point> dl_points = fld->DetectFaceLandmarks(path);

    for (unsigned long k = 0; k < dl_points.size(); ++k)
    {
        // cout << "pixel position of " <<k<<"  part: x= " << dl_points[k].x()<<" y= " <<dl_points[k].y()<< endl;
        points.push_back(Point2f(dl_points[k].x(), dl_points[k].y()));
    }

    InsertLastPoints(img,  points);
    // Insert points into subdiv
//    for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
//    {
//        subdiv.insert(*it);
//
//    }

   // vector<Point2f> im1_triangles = getVectorTriangles(img, subdiv);

    //Draw points

//    for (size_t i = 0; i < im1_triangles.size(); i = i + 3)
//    {
//        line(img, im1_triangles[i], im1_triangles[i + 1], delaunay_color, 1, CV_AA, 0);
//        line(img, im1_triangles[i], im1_triangles[i + 2], delaunay_color, 1, CV_AA, 0);
//        line(img, im1_triangles[i + 1], im1_triangles[i + 2], delaunay_color, 1, CV_AA, 0);
//    }
//

//    int m_i=0;
//    for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
//    {
//        draw_point(img, *it, points_color, std::to_string(m_i));
//        m_i++;
//    }

    //Show results.=============================================================================================================================
   // imshow(win_delaunay, img / 255);




    // Define colors for drawing.

    // Read in the image.
    Mat img1 = imread(path1);
    if(img.rows!=img1.rows||img.cols!=img1.cols)
    {
        cv::imwrite("tmp"+path1,img1);
        cv::resize(img1,img1,cv::Size(img.cols,img.rows));
        cv::imwrite(path1,img1);
    }

    img1.convertTo(img1, CV_32F);

    // Rectangle to be used with Subdiv2D
    Size size1 = img1.size();
    Rect rect1(0, 0, size.width, size.height);

    // Create a vector of points.
    vector<Point2f> points1;

    std::vector<dlib::point> dl_points1 = fld->DetectFaceLandmarks(path1);

    for (unsigned long k = 0; k < dl_points1.size(); ++k)
    {
        //cout << "pixel position of " <<k<<"  part: x= " << dl_points1[k].x()<<" y= " <<dl_points1[k].y()<< endl;
        points1.push_back(Point2f(dl_points1[k].x(), dl_points1[k].y()));
    }

    InsertLastPoints(img1,  points1);

    std::vector<Point2f> avgPoints;

    for (int i = 0; i < points.size(); i++)
    {
        float x, y;
        x =  (points[i].x +  points1[i].x)*0.5;
        y =  (points[i].y +  points1[i].y)*0.5;

        avgPoints.push_back(Point2f(x, y));
    }

    Mat img1Warped = img.clone();

    vector< vector<int> > dt;
    Rect rectWarped(0, 0, img1Warped.cols, img1Warped.rows);
    calculateDelaunayTriangles(rectWarped, avgPoints, dt);

   // imshow("dl1", img1/255);
    vector<Point2f> morphPoints;
    double alpha = 0.5;
    //empty average image
    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);
    //compute weighted average point coordinates
    for (int i = 0; i < points.size(); i++)
    {
        float x, y;
        x = (1 - alpha) * points[i].x + alpha * points1[i].x;
        y = (1 - alpha) * points[i].y + alpha * points1[i].y;

        morphPoints.push_back(Point2f(x, y));
    }


    for (int i = 0; i < dt.size(); i ++)
    {
        // Triangles
        vector<Point2f> t1, t2, t;


        for(size_t j = 0; j < 3; j++)
        {
            t1.push_back(points[dt[i][j]]);
            t2.push_back(points1[dt[i][j]]);
            t.push_back(morphPoints[dt[i][j]]);
        }

        morphTriangle(img, img1, imgMorph, t1, t2, t, alpha);

    }
//
//
     imwrite(path1+"Morphed_Face.jpg", imgMorph );
//


    return 0;
}