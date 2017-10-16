// feature.h 
// Tinh toan cac dac trung cua ki tu cho viec huan luyen va nhan dang
//..
//..

#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "dirent.h"

using namespace std;
using namespace cv;

const int number_of_feature = 32;

static int count_pixel(Mat img, bool black_pixel = true)
{
	int black = 0;
	int white = 0;
	for(int i = 0; i < img.rows; ++i)
		for(int j = 0; j < img.cols; ++j)
		{
				if(img.at<uchar>(i,j) == 0) 
					black++;
				else
					white++;
		}
	if(black_pixel) 
		return black;
	else
		return white;
}

static vector<float> calculate_feature(Mat src)
{
	Mat img;
	if(src.channels() == 3)
	{
		cvtColor(src, img, CV_BGR2GRAY);
		threshold(img, img, 100, 255, CV_THRESH_BINARY);
	}
	else
	{
		threshold(src, img, 100, 255, CV_THRESH_BINARY);
	}

	vector<float> r;
	//vector<int> cell_pixel;
	resize(img, img, Size(40, 40));
	int h = img.rows/4;
	int w = img.cols/4;
	int S = count_pixel(img);
	int T = img.cols * img.rows;
	for(int i = 0; i < img.rows; i += h)
	{
		for(int j = 0; j < img.cols; j += w)
		{
			Mat cell = img(Rect(i,j, h , w));
			int s = count_pixel(cell);
			float f = (float)s/S;
			r.push_back(f);
		}
	}
	
	for(int i = 0; i < 16; i+= 4)
	{
		float f = r[i] + r[i+1] + r[i+2] + r[i+3];
		r.push_back(f);
	}
	
	for(int i = 0; i < 4; ++i)
	{
		float f = r[i] + r[i+4] + r[i+8] + r[i+ 12];
		r.push_back(f);
	}
	
	r.push_back(r[0] + r[5] + r[10] + r[15]);
	r.push_back(r[3] + r[6] + r[9] + r[12]);
	r.push_back(r[0] + r[1] + r[4] + r[5]);
	r.push_back(r[2] + r[3] + r[6] + r[7]);
	r.push_back(r[8] + r[9] + r[12] + r[13]);
	r.push_back(r[10] + r[11] + r[14] + r[15]);
	r.push_back(r[5] + r[6] + r[9] + r[10]);
	r.push_back(r[0] + r[1] + r[2] + r[3] + r[4] + r[7] + r[8] + r[11] + r[12] + r[13] + r[14] + r[15]);

	return r; //32 feature
}


#endif