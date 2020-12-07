#include"utils.h"

void getAllFiles(string path, vector<string>& files)   ///windows系统里遍历文件夹里的全部文件和目录, 如果在ubuntu系统里，需要屏蔽这段代码
{
	intptr_t hFile = 0;//文件句柄  64位下long 改为 intptr_t
	struct _finddata_t fileinfo;//文件信息 
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1) //文件存在
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))//判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//文件夹名中不含"."和".."
				{
					//files.push_back(p.assign(path).append("/").append(fileinfo.name)); //保存文件夹名
					getAllFiles(p.assign(path).append("/").append(fileinfo.name), files); //递归遍历文件夹里的文件
				}
			}
			else
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));//如果不是文件夹，储存文件名
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/*   ///ubuntu系统里遍历文件夹里的全部文件和目录, 如果在windows系统里，需要屏蔽这段代码
void getAllFiles_u( const char * dir_name, vector<string>& files )
{
    // check the parameter !
    if( NULL == dir_name )
    {
        cout<<" dir_name is null ! "<<endl;
        return;
    }

    // check if dir_name is a valid dir
    struct stat s;
//    lstat( dir_name , &s );
//    if( ! S_ISDIR( s.st_mode ) )
//    {
//        cout<<dir_name<<" is not a valid directory !"<<endl;
//        return;
//    }

    struct dirent * filename;    // return value for readdir()
    DIR * dir;                   // return value for opendir()
    dir = opendir( dir_name );
//    if( NULL == dir )
//    {
//        cout<<"Can not open dir "<<dir_name<<endl;
//        return;
//    }
//    cout<<"Successfully opened the dir !"<<endl;

    /// read all the files in the dir ~
    chdir(dir_name);
    while( ( filename = readdir(dir) ) != NULL )
    {
        lstat(filename->d_name, &s);   // 获取下一级成员属性
        if(S_ISDIR(s.st_mode))  // 判断下一级成员是否是目录
        {
            // get rid of "." and ".."
            if( strcmp( filename->d_name , "." ) != 0 && strcmp( filename->d_name , "..") != 0)
            {
                char* abspath = new char[strlen(dir_name) + strlen(filename->d_name)+1];
                sprintf(abspath, "%s/%s", dir_name, filename->d_name);
                getAllFiles(abspath, files);
                delete [] abspath;
            }
        }
        else
        {
            string filepath = dir_name;
            filepath.append("/").append(filename->d_name);
            files.push_back(filepath);
        }
    }
    chdir( ".. ");
    closedir(dir);
}
*/
	
int write_face_feature_name2bin(int num_face, int len_feature, const float* output, const vector<string> names, const char* bin_name)
{
	FILE* fp = fopen(bin_name, "wb");
	fwrite(&num_face, sizeof(int), 1, fp);
	fwrite(&len_feature, sizeof(int), 1, fp);
	fwrite(output, sizeof(float), num_face * len_feature, fp);
	for (int i = 0; i < names.size(); i++)   //// num_face == names.size();
	{
		int len_s = names[i].length();
		fwrite(&len_s, sizeof(int), 1, fp);
		fwrite(names[i].c_str(), sizeof(char), len_s + 1, fp);   ///字符串末尾'\0'也算一个字符的
	}
	fclose(fp);
	return 0;
}

float* read_face_feature_name2bin(int* num_face, int* len_feature, vector<string>& names, const char* bin_name)
{
	FILE* fp = fopen(bin_name, "rb");
	fread(num_face, sizeof(int), 1, fp);
	fread(len_feature, sizeof(int), 1, fp);
	int len = (*num_face) * (*len_feature);
	float* output = new float[len];
	fread(output, sizeof(float), len, fp);//导入数据
	for (int i = 0; i < *num_face; i++)
	{
		int len_s = 0;
		fread(&len_s, sizeof(int), 1, fp);
		char* name = new char[len_s + 1];   ///字符串末尾'\0'也算一个字符的
		fread(name, sizeof(char), len_s + 1, fp);//导入数据
		//cout << name << endl;
		names.push_back(name);
		delete[] name;
	}

	fclose(fp);//关闭文件。
	return output;
}

int Get_Min_Euclid_Dist(float* face_features, vector<float> out_feature, int num_face, int len_feature, float* dist_feature) ////欧几里得距离值越小,两个向量越相似
{
	int i = 0, j = 0, min_ind = 0;
	float euclid_dist = 0, square = 0, min_dist = 10000;
	for (i = 0; i < num_face; i++)
	{
		euclid_dist = 0;
		for (j = 0; j < len_feature; j++)
		{
			square = (out_feature[j] - face_features[i * len_feature + j]) * (out_feature[j] - face_features[i * len_feature + j]);
			euclid_dist += square;
		}
		euclid_dist = sqrt(euclid_dist);
		dist_feature[i] = euclid_dist;
		if (euclid_dist < min_dist)
		{
			min_dist = euclid_dist;
			min_ind = i;
		}
	}
	return min_ind;
}

/*余弦距离值越大,两个向量越相似
* 定义向量a和向量b
余弦值cos(theta) = (a * b) / (|a| * |b|)
|a|和|b|表示向量a和向量b的模
在计算余弦值之前，已经对向量做了单位归一化，因此|a| = |b| = 1
那么cos(theta) = (a * b)
*/
int Get_Max_Cos_Dist(float* face_features, vector<float> out_feature, int num_face, int len_feature, float* dist_feature)   ////余弦距离值越大,两个向量越相似
{
	int i = 0, j = 0, max_ind = 0;
	float cos_dist = 0, max_dist = -10000;
	for (i = 0; i < num_face; i++)
	{
		cos_dist = 0;
		for (j = 0; j < len_feature; j++)
		{
			cos_dist = cos_dist + (out_feature[j] * face_features[i * len_feature + j]);
		}
		dist_feature[i] = cos_dist;
		if (cos_dist > max_dist)
		{
			max_dist = cos_dist;
			max_ind = i;
		}
	}
	return max_ind;
}
