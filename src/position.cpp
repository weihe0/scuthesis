#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "segment.hpp"
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
	////////////////////////////////////////////////////////////////////
        // cout << "label:" << static_cast<int>(it->first) << endl	  //
	//      << "dens:" << dens << endl				  //
	//      << "ratio:" << ratio << endl				  //
	//      << "left:" << it->second.left << endl			  //
	//      << "right:" << it->second.right << endl			  //
	//      << "up:" << it->second.up << endl			  //
	//      << "down:" << it->second.down << endl << endl;		  //
        ////////////////////////////////////////////////////////////////////
	if (dens > 0.5 && dens < 0.9 && ratio > 1.9 && ratio < 2.9 && it->second.pt > 5000)        
	{			        
	    digLabel = it->first;
	    rect = it->second;	        
	    break;			        
	}			        
    }
    /////////////////////////////////////////////////
    // cout << static_cast<int>(digLabel) << endl; //
    /////////////////////////////////////////////////
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

bool findWheel(Mat &src, Mat &bin, Mat &dst)
{
    Mat candidate;
    findComponents(bin, candidate);
    MatIterator_<uchar> it;
    for (it = candidate.begin<uchar>(); it != candidate.end<uchar>(); it++)
    {
	*it *= 10;
    }
    imwrite("candidate.png", candidate);
    Mat connect;
    struct Comp digitAttr;
    filterComponents(candidate, connect, digitAttr);
    vector<vector<Point> > contours;
    findContours(connect, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    Mat mask(bin.size(), bin.type(), Scalar(0));
    drawContours(mask, contours, 0, CV_RGB(255, 255, 255), CV_FILLED);
    Mat part;
    //src.copyTo(part, mask);
    dst = src(Range(digitAttr.up, digitAttr.down),
	      Range(digitAttr.left, digitAttr.right)).clone();
}

void refine(Mat &src, Mat &dst, Mat &mask)
{
    CV_Assert(src.size() == mask.size());
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(mask.type() == CV_8UC1);
    dst = src.clone();
    for (int r = 0; r < src.rows; r++)
    {
	for (int c = 0; c < src.cols; c++)
	{
	    if (mask.at<uchar>(r, c) == 0)
	    {
		dst.at<uchar>(r, c) = 0;
	    }
	}
    }
}
