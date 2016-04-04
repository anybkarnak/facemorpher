//
// Created by akomandyr on 28.03.16.
//

#ifndef FACEMORPHER_H
#define FACEMORPHER_H

#include "landmark_detector.h"

#include <string>

class FaceMorpher
{
public:
    int MorphFace(const std::string& src1, const std::string& src2, const std::string& dst);
private:

};
#endif //FACEMORPHER_FACEMORPHER_H
