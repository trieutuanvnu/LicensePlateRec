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
bool TrainSVM(string savepath, string trainImgpath) {
	const int number_of_class = 30;
	const int number_of_sample = 10;
	const int number_of_feature = 32;

	//Train SVM OpenCV 3.1
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(0.5);
	svm->setC(16);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	vector<string> folders = list_folder(trainImgpath);
	if (folders.size() <= 0)
	{
		//do something
		return false;
	}
	if (number_of_class != folders.size() || number_of_sample <= 0 || number_of_class <= 0)
	{
		//do something
		return false;
	}
	Mat src;
	Mat data = Mat(number_of_sample * number_of_class, number_of_feature, CV_32FC1);
	Mat label = Mat(number_of_sample * number_of_class, 1, CV_32SC1);
	int index = 0;
	std::sort(folders.begin(),folders.end());
	for (size_t i = 0; i < folders.size(); ++i)
	{
		vector<string> files = list_file(folders.at(i));
		if (files.size() <= 0 || files.size() != number_of_sample)
		{
			return false;
		}
		string folder_path = folders.at(i);
		cout << "list folder" << folders.at(i) << endl;
		string label_folder = folder_path.substr(folder_path.length() - 1);
		for (size_t j = 0; j < files.size(); ++j)
		{
			src = imread(files.at(j));
			if (src.empty())
			{
				return false;
			}
			vector<float> feature = calculate_feature(src);
			for (size_t t = 0; t < feature.size(); ++t)
				data.at<float>(index, t) = feature.at(t);
			label.at<int>(index, 0) = i;
			index++;
		}
	}
	// SVM Train OpenCV 3.1
	svm->trainAuto(ml::TrainData::create(data, ml::ROW_SAMPLE, label));
	svm->save(savepath);
	return true;
}

int main(int argc, char* argv[]) {

	std::string savesvm = "svm.txt";
	std::string imgpath = "./data/";
	bool train = TrainSVM(savesvm, imgpath);
	if (train)
	{
		cout << "SVM Training Completed" << endl;
	}
	else
		cout << " Train ERROR " << endl;
	cout << "Press any key to exit." << endl;
	getwchar();
	return 0;

}
