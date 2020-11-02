#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <io.h>
#include <iostream>
#include <string.h>
#include<fstream>

using namespace std;
/*这个文件夹里包含多个子文件夹, 每个子文件夹的名字是人名. 
子文件夹里包含这个人的人脸图片, 
图片是肖像照, 里面只有一个人脸*/
void getAllFiles(string path, vector<string>& files);  ///windows系统里遍历文件夹里的全部文件和目录

inline int MinInt(int a, int b)    //返回整数a和b中较小的一个
{
	return (a < b) * a + (1 - (a < b)) * b;
}

inline int MaxInt(int a, int b)    //返回整数a和b中较大的一个
{
	return (a > b) * a + (1 - (a > b)) * b;
}

inline string fromPath_Getname(string filepath)
{
	size_t pos_end = filepath.rfind("/");    ////倒数第1个路径间隔符， 路径分隔符是/
	size_t pos_start = filepath.substr(0, pos_end).rfind("/");   ////倒数第2个路径间隔符
	return filepath.substr(pos_start + 1, pos_end - pos_start - 1);
}

inline string fromPath_Get_imgname(string filepath)
{
	size_t pos_end = filepath.rfind("/");    ////倒数第1个路径间隔符
	return filepath.substr(pos_end + 1);
}

int write_face_feature_name2bin(int num_face, int len_feature, const float* output, const vector<string> names, const char* bin_name);

float* read_face_feature_name2bin(int* num_face, int* len_feature, vector<string>& names, const char* bin_name);

int Get_Min_Euclid_Dist(float* face_features, vector<float> out_feature, int num_face, int len_feature, float* dist_feature);

int Get_Max_Cos_Dist(float* face_features, vector<float> out_feature, int num_face, int len_feature, float* dist_feature);

typedef struct Rec_thresh
{
	float thresh;
	string type;
} Rec_thresh;