#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include "position.hpp"
#include "unionfind.hpp"
using namespace std;

void Comp::insert(int r, int c)
{
    if (pt == 0)
    {
	up = down = r;
	left = right = c;
    }
    else
    {
	if (r < up)
	{
	    up = r;
	}
	else if (r > down)
	{
	    down = r;
	}

	if (c < left)
	{
	    left = c;
	}
	else if (c > right)
	{
	    right = c;
	}
    }
    pt++;
}

void findComponents(Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_8UC1);
    Mat tmp(src.size(), src.type());
    uchar label = 1;
    UnionFindSet ufSet(200);
    for (int r = 0; r < src.rows; r++)
    {
	for (int c = 0; c < src.cols; c++)
	{
	    if (src.at<uchar>(r, c) == 0)
	    {
		tmp.at<uchar>(r, c) = 0;
	    }
	    else
	    {
		size_t nb = 0;
		uchar nbLabel[2];
		if (r >= 1 && tmp.at<uchar>(r - 1, c) != 0)
		{
		    nbLabel[nb] = tmp.at<uchar>(r - 1, c);
		    nb++;
		}
		if (c >= 1 && tmp.at<uchar>(r, c - 1) != 0)
		{
		    nbLabel[nb] = tmp.at<uchar>(r, c - 1);
		    nb++;
		}
		switch (nb)
		{
		case 0:
		    tmp.at<uchar>(r, c) = label;
		    label++;
		    break;
		case 1:
		    tmp.at<uchar>(r, c) = nbLabel[0];
		    break;
		case 2:
		    tmp.at<uchar>(r, c) = min(nbLabel[0], nbLabel[1]);
		    if (nbLabel[0] != nbLabel[1])
		    {
			ufSet.unionSet(nbLabel[0], nbLabel[1]);
		    }
		    break;
		}
	    }
	}
    }

    for (int r = 0; r < tmp.rows; r++)
    {
	for (int c = 0; c < tmp.cols; c++)
	{
	    if (src.at<uchar>(r, c) != 0)
	    {
		tmp.at<uchar>(r, c) = ufSet.findSet(tmp.at<uchar>(r, c));
	    }
	}
    }
    dst = tmp;
     
}

bool filterComponents(Mat &src, Mat &dst, struct Comp &rect)
{
    CV_Assert(src.type() == CV_8UC1);
    dst.create(src.size(), src.type());
    map<uchar, struct Comp> compMap;
    for (int r = 0; r < src.rows; r++)
    {
	for (int c = 0; c < src.cols; c++)
	{
	    uchar label = src.at<uchar>(r, c);
	    if (label != 0)
	    {
		if (compMap.find(label) == compMap.end())
		{
		    compMap[label] = Comp();
		}
		compMap[label].insert(r, c);
	    }
	}
    }

    map<uchar, struct Comp>::iterator it, itend;
    itend = compMap.end();
    uchar digLabel = 0;
    for (it = compMap.begin(); it != itend; it++)
    {
	int width = it->second.right - it->second.left;
	int length = it->second.down - it->second.up;
	double dens = static_cast<double>(it->second.pt) / (width * length);
	double ratio = static_cast<double>(width) / length;
	if (dens > 0.5 && dens < 0.9 && ratio > 1.9 && ratio < 2.9 && it->second.pt > 5000)        
	{			        
	    digLabel = it->first;
	    rect = it->second;	        
	    break;			        
	}			        
    }
    for (int r = 0; r < dst.rows; r++)
    {
	for (int c = 0; c < dst.cols; c++)
	{
	    if (src.at<uchar>(r, c) == digLabel)
	    {
		dst.at<uchar>(r, c) = digLabel;
	    }
	}
    }
    return digLabel != 0;
}

int selectContour(vector<vector<Point> > &contours)
{
    int idx;
    Mat rotated;
    for (idx = 0; idx < contours.size(); idx++)
    {
	Rect rect = boundingRect(contours[idx]);
	double area = contourArea(contours[idx]);
	double ratio = static_cast<double>(rect.width) / rect.height;
	double rho = area / (rect.width * rect.height);
	if ((ratio > 1.5 && ratio < 3.0)
	    && (rho > 0.5 && rho < 1.0))
	{
	    break;
	}
    }
    return (idx >= contours.size()) ? -1 : idx;
}

double slantAngle(vector<Point> &frame)
{
    double epsilon = frame.size() * 0.05;
    vector<Point> poly;
    approxPolyDP(frame, poly, epsilon, true);
    int top = 0;
    for (int i = 1; i < poly.size(); i++)
    {
	if (poly[i].y < poly[top].y)
	{
	    top = i;
	}
    }
    int prev = (top - 1 < 0) ? poly.size() - 1 : top - 1;
    int next = (top + 1) % poly.size();
    double theta;
    if (abs(poly[top].x - poly[prev].x) > 50)
    {
	theta = atan(static_cast<double>
		       (poly[top].y - poly[prev].y)
		       / (poly[top].x - poly[prev].x));
    }
    else
    {
	theta = atan(static_cast<double>
		       (poly[top].y - poly[next].y)
		       / (poly[top].x - poly[next].x));
    }
    return theta * 180 / CV_PI;
}

bool findWheel(Mat &src, Mat &dst)
{
    Mat bin, binInv;
    threshold(src, bin, 60.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    imwrite("otsu.png", bin);
    imshow("otsu", bin);
    threshold(src, binInv, 60.0, 255.0, THRESH_BINARY_INV | THRESH_OTSU);

    Mat comp;
    findComponents(binInv, comp);
    for (int i = 0; i < comp.rows; i++)
    {
	for (int j = 0; j < comp.cols; j++)
	{
	    comp.at<uchar>(i, j) *= 10;
	}
    }
    imwrite("candidate.png", comp);

    vector<vector<Point> > contours;
    findContours(binInv, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int idx = selectContour(contours);

    if (idx < 0)
    {
	return false;
    }

    Rect rect = boundingRect(contours[idx]);
    Mat mask(bin.size(), bin.type(), Scalar(0));
    drawContours(mask, contours, idx, Scalar(255), CV_FILLED);
    Mat left;
    src.copyTo(left, mask);
    Mat rotated = src(rect).clone();

    HoughImg(rotated);

    double slant = slantAngle(contours[idx]);
    Point center;
    center.x = rotated.cols / 2;
    center.y = rotated.rows / 2;
    Mat rot = getRotationMatrix2D(center, slant, 1.0);

    warpAffine(rotated, dst, rot, rotated.size());    
    imwrite("frame.png", rotated);
    imwrite("rotate.png", dst);

    return true;
}
    

void HoughImg(Mat &src)
{
    Mat edge;
    Canny(src, edge, 65.0, 70.0);
    imwrite("canny.png", edge);
    vector<Vec4i> lines;
    Mat lineImg(src.size(), CV_8UC1, Scalar(0));
    HoughLinesP(edge, lines, 1, CV_PI / 180, 20);
    for (int i = 0; i < lines.size(); i++)
    {
	line(lineImg, Point(lines[i][0], lines[i][1]),
	     Point(lines[i][2], lines[i][3]), Scalar(255), 1);
    }
    imwrite("hough.png", lineImg);
}

