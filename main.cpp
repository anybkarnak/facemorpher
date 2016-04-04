#include <iostream>
#include "movie_creator.h"

using namespace std;

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
    return 0;
}