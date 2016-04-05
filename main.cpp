#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "movie_creator.h"
#include "landmark_detector.h"

using namespace std;
using namespace cv;


// Draw a single point
static void draw_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}

// Draw delaunay triangles
static void draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
{

    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}



int main(int argc,      // Number of strings in array argv
         char *argv[])  // Array of command-line argument strings
{
    int count;
// Display each command-line argument.
    cout << "\nCommand-line arguments:\n";
    for( count = 0; count < argc; count++ )
        cout << "  argv[" << count << "]   "
        << argv[count] << "\n";
    cout << "Hello, World!" << endl;

    std::string path = "ant.jpg";

    FaceLandmarkDetector* fld = new FaceLandmarkDetector("shape_predictor_68_face_landmarks.dat");
    std::vector<dlib::point> dl_points = fld->DetectFaceLandmarks(path);




// Define window names
    string win_delaunay = "Delaunay Triangulation";


    // Turn on animation while drawing triangles
    bool animate = true;

    // Define colors for drawing.
    Scalar delaunay_color(255,255,255), points_color(0, 0, 255);

    // Read in the image.
    Mat img = imread(path);

    // Keep a copy around
    Mat img_orig = img.clone();

    // Rectangle to be used with Subdiv2D
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Create a vector of points.
    vector<Point2f> points;

    for (unsigned long k = 0; k < dl_points.size(); ++k)
    {
        cout << "pixel position of " <<k<<"  part: x= " << dl_points[k].x()<<" y= " <<dl_points[k].y()<< endl;
        points.push_back(Point2f(dl_points[k].x(),dl_points[k].y()));
    }

    ///insert last points
    points.push_back(Point2f(1,1));
    points.push_back(Point2f(img.cols-1,img.rows-1));

    points.push_back(Point2f(1,img.rows-1));
    points.push_back(Point2f(img.cols-1,1));

    points.push_back(Point2f(1,img.rows/2));
    points.push_back(Point2f(img.cols/2,1));

    points.push_back(Point2f(img.cols/2,img.rows-1));
    points.push_back(Point2f(img.cols-1,img.rows/2));
    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
        // Show animation
        if (animate)
        {
            Mat img_copy = img_orig.clone();
            // Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, delaunay_color );
            imshow(win_delaunay, img_copy);
            waitKey(100);
        }

    }

    // Draw delaunay triangles
    draw_delaunay( img, subdiv, delaunay_color );

    // Draw points
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        draw_point(img, *it, points_color);
    }


    // Show results.
    imshow( win_delaunay, img);
    waitKey(0);

    return 0;
}