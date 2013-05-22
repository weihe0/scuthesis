#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include "position.hpp"

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

void drawProject(MatND &pro, Mat &proImg)
{
    double minVal, maxVal;
    minMaxLoc(pro, &minVal, &maxVal, 0, 0);
    int width = static_cast<int>(maxVal / 0.9);
    int length = pro.total();
    proImg.create(Size(length, width), CV_8UC1);
    MatIterator_<uchar> it;
    for (it = proImg.begin<uchar>(); it != proImg.end<uchar>(); it++)
    {
	*it = 255;
    }
    for (int h = 0; h < pro.total(); h++)
    {
	line(proImg, Point(h, width), 
	     Point(h, width - static_cast<int>(pro.at<float>(h))),
	     Scalar::all(0));
    }
}

void projectHor(Mat &src, MatND &dst, int pixel = 1)
{
    CV_Assert(src.type() == CV_8UC1);
    dst = Mat(1, &src.cols, CV_32F, Scalar::all(0));
    for (int j = 0; j < src.cols; j++)
    {
	for (int i = 0; i < src.rows; i++)
	{
	    if (((pixel != 0) && (src.at<uchar>(i, j) != 0))
		|| ((pixel == 0) && (src.at<uchar>(i, j) == 0)))
	    {
		++dst.at<float>(j);
	    }
	}
    }
}
		
	    
void projectVer(Mat &src, MatND &dst, int pixel = 1)
{	    
    CV_Assert(src.type() == CV_8UC1);
    dst = Mat(1, &src.rows, CV_32F, Scalar::all(0));
    for (int i = 0; i < src.rows; i++)
    {
	for (int j = 0; j < src.cols; j++)
	{
	    if (((pixel != 0) && (src.at<uchar>(i, j) != 0))
		|| ((pixel == 0) && (src.at<uchar>(i, j) == 0)))
	    {
		++dst.at<float>(i);
	    }
	}
    }
}


void otsuSig( const Mat &_src, MatND &sigArr)
{
    
    Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }
    const int N = 256;
    int i, j, h[N] = {0};
    int len[1];
    len[0] = N;
    sigArr = Mat(1, len, CV_32F, Scalar::all(0));
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.data + _src.step*i;
        j = 0;
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
	sigArr.at<float>(i) = static_cast<float>(sigma);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }
}

void diffOper(MatND &src, MatND &dst)
{
    CV_Assert(src.type() == CV_32F);
    int lenDim[0];
    lenDim[0] = src.total();
    Mat tmp(1, lenDim, CV_32F);
    for (int i = 0; i < src.total() - 1; i++)
    {
	tmp.at<float>(i) = src.at<float>(i + 1) - src.at<float>(i) + 50;
    }
    tmp.at<float>(tmp.total() - 1) = src.at<float>(src.total() - 1);
    dst = tmp.clone();
}

double percentBlack(Mat &bin)
{
    CV_Assert(bin.type() == CV_8UC1);
    int count = 0;
    MatIterator_<uchar> it;
    for (it = bin.begin<uchar>(); it != bin.end<uchar>(); it++)
    {
	if (*it == 0)
	{
	    count++;
	}
    }
    return static_cast<double>(count) / (bin.rows * bin.cols);
}



void notImg(Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_8UC1);
    dst.create(src.size(), src.type());
    for (int i = 0; i < src.rows; i++)
    {
	for (int j = 0; j < src.cols; j++)
	{
	    dst.at<uchar>(i, j) = src.at<uchar>(i, j) == 0 ? 255 : 0;
	}
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
 
    MatND hist;
    Mat histImg;
    getHist(gray, hist);
    drawHist(hist, histImg);
    //imwrite("histsrc.png", histImg);
    //imshow("hist", histImg);
    // equalizeHist(gray, gray);

    // getHist(gray, hist);
    // drawHist(hist, histImg);
    // imwrite("histequal.png", histImg);
    // imshow("histequal", histImg);
    dst = gray;
}

void findPeaks(MatND &pro, vector<Vec2i> &peaks, int minLen = 10)
{
    int s = 0;
    int e = 0;
    int state = 0;

    for (int i = 0; i < pro.total(); i++)
    {
	switch (state)
	{
	case 0:
	    if (pro.at<float>(i) > FLT_EPSILON)
	    {
		s = e = i;
		state = 1;
	    }
	    break;

	case 1:
	    if (pro.at<float>(i) < FLT_EPSILON)
	    {
		e = i;
		if (e - s > minLen)
		{
		    peaks.push_back(Vec2i(s, e));
		}
		s = e = i;
		state = 0;
	    }
	}
    }
    if (state == 1)
    {
	e = pro.total();
	if (e - s > minLen )
	{
	    peaks.push_back(Vec2i(s, e));
	}
    }
}

void segment(Mat &src, Mat &dst)
{
    adaptiveThreshold(src, dst, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C,
		      3, THRESH_BINARY, 1.0);
    imshow("dst", src);
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
	return -1;
    }
    Mat src = imread(argv[1]);
    imshow("src", src);
    if (src.data == NULL)
    {
	cerr << "cannot open " << argv[1] << endl;
	return -1;
    }
    Mat gray;
    preprocess(src, gray);
    Mat wheel;
    findWheel(gray, wheel);
    imshow("wheel", wheel);

    MatND px;
    Mat pxImg;
    projectHor(wheel, px, 1);
    drawProject(px, pxImg);
    imwrite("projectx.png", pxImg);
    imshow("horizontal", pxImg);

    vector<Vec2i> horPeaks;
    findPeaks(px, horPeaks, 10);
    vector<Mat> horCut;
    const char *names[6] = {"1.png", "2.png", "3.png", "4.png", "5.png"};
    for (int i = 0; i < horPeaks.size(); i++)
    {
	Mat candDigit = wheel(Range(0, wheel.rows),
			Range(horPeaks[i][0], horPeaks[i][1])).clone();
	horCut.push_back(candDigit);
	MatND py;
	vector<Vec2i> verPeaks;
	projectVer(wheel, py, 1);
	findPeaks(py, verPeaks, 20);
	Mat digit = candDigit(Range(verPeaks[0][0], verPeaks[0][1]),
			      Range(0, candDigit.cols));
	//imwrite(names[i], digit);
	imshow(names[i], digit);
    }
    waitKey(0);
}



