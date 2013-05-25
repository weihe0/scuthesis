#ifndef POSITION_H
#define POSITION_H
#include <opencv2/core/core.hpp>
using namespace cv;
struct Comp
{
    int pt;
    int up;
    int down;
    int left;
    int right;
    Comp() : pt(0), left(0), right(0), up(0), down(0) {}
    void insert(int r, int c);
};


void  fixedThreshold(Mat &src, Mat &dst, uchar threshold);
void findComponents(Mat &src, Mat &dst);
bool filterComponents(Mat &src, Mat &dst, struct Comp &rect);
void refine(Mat &src, Mat &dst, Mat &mask);
bool findWheel(Mat &src, Mat &dst);
void calibrate(Mat &src, Mat &dst);
void HoughImg(Mat &src);

#endif

