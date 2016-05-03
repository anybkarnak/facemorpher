#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "movie_creator.h"

using namespace std;

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
         path = "Angelina.jpg";
         path1 = "sasha2.jpg";
    }


    FaceMorpherPtr morpherPtr = std::make_shared<FaceMorpher>();

    morpherPtr->MorphFace(path, path1, "by_class_morphed"+path);

    return 0;
}