###################################################
##Copyright by trieutuanvn/jackyle#################
###################################################

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include "dirent.h"
#include "feature.h"
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::ml;
Mat srcImg;


vector<string> list_folder(string path)
{
	vector<string> folders;
	DIR *dir = opendir(path.c_str());
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL)
	{
		if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0))
		{
			string folder_path = path + "/" + string(entry->d_name);
			folders.push_back(folder_path);
		}
	}
	closedir(dir);
	return folders;

}
vector<string> list_file(string folder_path)
{
	vector<string> files;
	DIR *dir = opendir(folder_path.c_str());
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL)
	{
		if ((strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name, "..") != 0))
		{
			string file_path = folder_path + "/" + string(entry->d_name);
			files.push_back(file_path);
		}
	}
	closedir(dir);
	return files;
}
char character_recognition(Mat img_character)
{
	//Load SVM training file OpenCV 3.1
	Ptr<SVM> svmNew = SVM::create();
	svmNew = SVM::load("svm.txt");
	char c = '*';

	vector<float> feature = calculate_feature(img_character);
	// Open CV3.1
	Mat m = Mat(1, number_of_feature, CV_32FC1);
	for (size_t i = 0; i < feature.size(); ++i)
	{
		float temp = feature[i];
		m.at<float>(0, i) = temp;
	}

	int ri = int(svmNew->predict(m)); // Open CV 3.1
									  /*int ri = int(svmNew.predict(m));*/
	if (ri >= 0 && ri <= 9)
		c = (char)(ri + 48); //ma ascii 0 = 48
	if (ri >= 10 && ri < 18)
		c = (char)(ri + 55); //ma accii A = 5, --> tu A-H
	if (ri >= 18 && ri < 22)
		c = (char)(ri + 55 + 2); //K-N, bo I,J
	if (ri == 22) c = 'P';
	if (ri == 23) c = 'S';
	if (ri >= 24 && ri < 27)
		c = (char)(ri + 60); //T-V,  
	if (ri >= 27 && ri < 30)
		c = (char)(ri + 61); //X-Z

	return c;

}
string SVMPredict() {
	string licenseRecog;
	vector<Mat> plates;
	vector<Mat> draw_character;
	vector<vector<Mat> > characters;
	vector<string> text_recognition;
	vector<double> process_time;
	void sort_character(vector<Mat>&);
	Mat image = srcImg;
	Mat gray, binary;
	vector<vector<cv::Point> > contours;
	vector<Vec4i> hierarchy;
	double t = (double)cvGetTickCount();
	cvtColor(image, gray, CV_BGR2GRAY);
	//imshow("gray", gray);
	adaptiveThreshold(gray, binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 55, 5);
	//imshow("binary",binary);
	Mat or_binary = binary.clone();
	Mat element = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
	erode(binary, binary, element);
	dilate(binary, binary, element);
	findContours(binary, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	if (contours.size() <= 0) return false;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		Rect r = boundingRect(contours.at(i));
		if (r.width > image.cols / 2 || r.height > image.cols / 2 || r.width < 120 || r.height < 20
			|| (double)r.width / r.height > 4.5 || (double)r.width / r.height < 3.5)
			continue;
		Mat sub_binary = or_binary(r);
		Mat _plate = sub_binary.clone();
		vector<vector<cv::Point> > sub_contours;
		vector<Vec4i> sub_hierarchy;
		findContours(sub_binary, sub_contours, sub_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		if (sub_contours.size() < 8) continue;
		int count = 0;
		vector<Mat> c;
		Mat sub_image = image(r);
		vector<Rect> r_characters;
		for (size_t j = 0; j < sub_contours.size(); ++j)
		{
			Rect sub_r = boundingRect(sub_contours.at(j));
			if (sub_r.height > r.height / 2 && sub_r.width < r.width / 8 && sub_r.width > 5 && r.width > 15 && sub_r.x > 5)
			{
				Mat cj = _plate(sub_r);
				double ratio = (double)count_pixel(cj) / (cj.cols*cj.rows);
				if (ratio > 0.2 && ratio < 0.7)
				{
					r_characters.push_back(sub_r);
					rectangle(sub_image, sub_r, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
		}
		if (r_characters.size() >= 7)
		{
			// sap xep 
			for (int i = 0; i < r_characters.size() - 1; ++i)
			{
				for (int j = i + 1; j < r_characters.size(); ++j)
				{
					Rect temp;
					if (r_characters.at(j).x < r_characters.at(i).x)
					{
						temp = r_characters.at(j);
						r_characters.at(j) = r_characters.at(i);
						r_characters.at(i) = temp;
					}
				}
			}
			for (int i = 0; i < r_characters.size(); ++i)
			{
				Mat cj = _plate(r_characters.at(i));
				c.push_back(cj);
			}
			characters.push_back(c);
			sub_binary = or_binary(r);
			plates.push_back(_plate);
			draw_character.push_back(sub_image);
		}
		rectangle(image, r, Scalar(0, 255, 0), 2, 8, 0);
	}
	imshow("place",image);
	imshow("char", draw_character[0]);
	// Plate recoginatinon
	for (size_t i = 0; i < characters.size(); i++)
	{
		string result;
		for (size_t j = 0; j < characters.at(i).size(); ++j)
		{

			char cs = character_recognition(characters.at(i).at(j));
			result.push_back(cs);

		}
		text_recognition.push_back(result);
		licenseRecog += result;
	}
	return licenseRecog;
}



int main(int argc, char* argv[]) {
	srcImg = imread("./test/0008.JPG");
	cout << " Car License Deteceted Number: " << SVMPredict() <<endl;
	waitKey(0);
	cout << "Press any key to exit." << endl;
	getwchar();
	return 0;

}
