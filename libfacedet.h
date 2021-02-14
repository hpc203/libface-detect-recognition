#include "opencv2/opencv.hpp"
#include"utils.h"
using namespace cv;
using namespace dnn;
//using namespace std;

// center: [x, y], width: w, height: h
typedef struct BndBox_xywh {
    Point2f center;
    float w;
    float h;
} BndBox_xywh;

// top_left: [x1, y1], bottom_right: [x2, y2]
typedef struct BndBox_xyxy {
    Point2f top_left;
    Point2f bottom_right;
    float area() {
        return (bottom_right.x - top_left.x + 1) * (bottom_right.y - top_left.y + 1);
    }
} BndBox_xyxy;

typedef struct Landmarks_10 {
    // right eye
    Point2f right_eye;
    // left eye
    Point2f left_eye;
    // mouth left
    Point2f mouth_left;
    // nose
    Point2f nose_tip;
    // mouth right
    Point2f mouth_right;
} Landmarks_10;

typedef struct Face {
    BndBox_xyxy bbox;
    Landmarks_10 landmarks;
    float score;
} Face;

class PriorBox 
{
    private:
        const vector<vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        const vector<int> steps = { 8, 16, 32, 64 };
        const vector<float> variance = { 0.1, 0.2 };

        int in_w;
        int in_h;
        vector<Size> feature_map_sizes;
        vector<BndBox_xywh> priors;
    private:
        vector<BndBox_xywh> generate_priors();
    public:
        PriorBox(const int width);
        ~PriorBox();
        vector<Face> decode(const Mat& loc, const Mat& conf, const Size output_shape);
};

class libface
{
    public:
        libface(bool align=false, int width = 320) :pb(width)
        {
            this->align = align;
			string modelpath = "models/YuFaceDetectNet_" + to_string(width) + ".onnx";  ///"_320.onnx"
			this->model_path = modelpath;
            this->net = readNet(model_path);
            this->input_shape = Size(width, int(0.75 * width));
        }
        vector<Face> detect(Mat img);
        Mat crop_face(Face det, Mat srcimg);
        void draw(Mat& img, const vector<Face>& faces, bool is_draw_lanmarks);
    private:
        Net net;
        string model_path;
        bool align;
        const float conf_thresh = 0.6;
        const float nms_thresh = 0.3;
        const int keep_top_k = 750;
        const vector<String> output_names = { "loc", "conf" };
        Size get_input_shape(string model_fpath);
        Size input_shape;
		PriorBox pb;
        void nms(vector<Face>& dets, const float thresh);
};

class openface    ////参考 https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
{
    public:
        openface()
        {
            this->net = readNet(this->model_path);
        }
        int get_feature_length()
        {
            return this->length;
        }
        vector<float> get_feature(Mat img);
    private:
        Net net;
        const string model_path = "models/openface_nn4.small2.v1.t7";
        const int inpWidth = 96;
        const int inpHeight = 96;
        const int length = 128;
};

class arcface   ///我自己把pytorch的.pth文件转换到onnx文件，这样dnn就能读取了
{
    public:
        arcface()
        {
            this->net = readNetFromONNX(this->model_path);
        }
        int get_feature_length()
        {
            return this->length;
        }
        vector<float> get_feature(Mat img);
    private:
        Net net;
        const string model_path = "models/resnet18_110.onnx";
        const int inpWidth = 128;
        const int inpHeight = 128;
        const int length = 512;
};

class pfld    ///我自己把pytorch的.pth文件转换到onnx文件，这样dnn就能读取了
{
    public:
        pfld()
        {
            this->net = readNetFromONNX(this->model_path);
        }
        vector<Point> detect(Mat crop_img);
        void draw_landmarks(vector<Point> pts, Mat& crop_img);
        void face_detect_draw_landmarks(vector<Point> pts, Mat& img, int start_x, int start_y);
    private:
        Net net;
        const string model_path = "models/pfld.onnx";
        const int inpWidth = 112;
        const int inpHeight = 112;
        const vector<String> output_names = { "output", "landmarks" };
};
