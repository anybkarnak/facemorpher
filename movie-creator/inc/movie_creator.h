//
// Created by akomandyr on 29.03.16.
//

#ifndef MOVIE_CREATOR_H
#define MOVIE_CREATOR_H

#include "face_morpher.h"

class MovieCreator
{
public:
    MovieCreator();
    int CreateMovie(const std::string& src1, const std::string& src2, const std::string& dst);
};

#endif //MOVIE_CREATOR_H
