#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "position.hpp"
#include "segment.hpp"
#include "recognise.hpp"
using namespace cv;
using namespace std;

void getHist(Mat &img, MatND &hist)
{
    int histSize[1];
    float hranges[2];
    const float *ranges[1];
    int channels[1];
    histSize[0] = 256;
    hranges[0] = 0.0;
    hranges[1] = 255.0;
    ranges[0] = hranges;
    channels[0] = 0;
    calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges);
}

void drawHist(MatND &hist, Mat &histImg)
{
    double maxVal = 0;
    double minVal = 0;
    int histSize[1];
    histSize[0] = 256;
    minMaxLoc(hist, &minVal, &maxVal, 0, 0);
    histImg.create(Size(histSize[0], histSize[0]), CV_8UC1);
    MatIterator_<uchar> it;
    for (it = histImg.begin<uchar>(); it != histImg.end<uchar>(); it++)
    {
	*it = 255;
    }
    int hpt = static_cast<int>(0.9 * histSize[0]);
    for (int h = 0; h < histSize[0]; h++)
    {
	float binVal = hist.at<float>(h);
	int intensity = static_cast<int>(binVal*hpt/maxVal);
	line(histImg, Point(h, histSize[0]), Point(h, histSize[0] - intensity)
	     ,Scalar::all(0));
    }
}
		



void preprocess(Mat &src, Mat &dst)
{
    Mat upright = src(Range(0, src.rows / 2),     
		       Range(src.cols / 2, src.cols));
    Mat gray;
    imwrite("src.png", src);
    imwrite("rgb.png", upright);
    cvtColor(upright, gray, CV_RGB2GRAY);
    imwrite("gray.png", gray);


    GaussianBlur(gray, gray, Size(3, 3), 1.0);
    imwrite("gaussian.png", gray);
 
    Mat gray2 = gray.clone();
    MatND hist;
    Mat histImg;
    getHist(gray2, hist);
    drawHist(hist, histImg);
    imwrite("histsrc.png", histImg);
    //    imshow("hist", histImg);
    equalizeHist(gray2, gray2);
    imwrite("equal.png", gray2);
    getHist(gray2, hist);
    drawHist(hist, histImg);
    imwrite("histequal.png", histImg);
    //    imshow("histequal", histImg);

    dst = gray;
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
	return -1;
    }
    Mat src = imread(argv[1]);
    //imshow("src", src);
    if (src.data == NULL)
    {
	cerr << "cannot open " << argv[1] << endl;
	return -1;
    }

    Mat gray;
    preprocess(src, gray);

    Mat wheel;
    findWheel(gray, wheel);
    imwrite("frame2.png", wheel);

    vector<Mat> digitVec;
    segment(wheel, digitVec);

    //Recogniser rec;

    const char *names[5] = {"1.png", "2.png", "3.png", "4.png", "5.png"};
    for (int i = 0; i < digitVec.size(); i++)
    {
      //imshow(names[i], digitVec[i]);
	imwrite(names[i], digitVec[i]);
	//rec.refine(digitVec[i], digitVec[i]);
	//imshow(names[i], digitVec[i]);
    }
    waitKey(0);
}
