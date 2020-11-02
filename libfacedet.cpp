#include"libfacedet.h"

void Min_Max_Normalization(vector<float>& output)
{
    float min = *min_element(output.begin(), output.end());//返回s中的最小值
    float max = *max_element(output.begin(), output.end());//返回最大值
    float s = 1.0 / (max - min);
    for (int i = 0; i < output.size(); i++)
    {
        output[i] = (output[i] - min) * s;
    }
}

void Unit_Normalization(vector<float>& output)   //向量单位归一化
{
    float length = 0;
    for (int i = 0; i < output.size(); i++)
    {
        length = length + output[i] * output[i];
    }
    length = sqrt(length);
    float s = 1.0 / length;
    for (int i = 0; i < output.size(); i++)
    {
        output[i] *= s;
    }
}

PriorBox::PriorBox(const Size& input_shape,
    const Size& output_shape) {
    // initialize
    in_w = input_shape.width;
    in_h = input_shape.height;
    out_w = output_shape.width;
    out_h = output_shape.height;

    Size feature_map_2nd = {
        int(int((in_w + 1) / 2) / 2), int(int((in_h + 1) / 2) / 2)
    };
    Size feature_map_3rd = {
        int(feature_map_2nd.width / 2), int(feature_map_2nd.height / 2)
    };
    Size feature_map_4th = {
        int(feature_map_3rd.width / 2), int(feature_map_3rd.height / 2)
    };
    Size feature_map_5th = {
        int(feature_map_4th.width / 2), int(feature_map_4th.height / 2)
    };
    Size feature_map_6th = {
        int(feature_map_5th.width / 2), int(feature_map_5th.height / 2)
    };

    // feature_map_sizes.push_back(feature_map_2nd);
    feature_map_sizes.push_back(feature_map_3rd);
    feature_map_sizes.push_back(feature_map_4th);
    feature_map_sizes.push_back(feature_map_5th);
    feature_map_sizes.push_back(feature_map_6th);

    // cout << "Before priors gen!" << endl;
    priors = generate_priors();
    // cout << "Finished priors gen!" << endl;
}

PriorBox::~PriorBox() {}

vector<BndBox_xywh> PriorBox::generate_priors() {
    // cout << "In!" << endl;
    vector<BndBox_xywh> anchors;
    // cout << "Before loop!" << endl;
    // cout << feature_map_sizes.size() << endl;
    for (auto i = 0; i < feature_map_sizes.size(); ++i) {
        // cout << "In for loop! " << i << endl;
        Size feature_map_size = feature_map_sizes[i];
        vector<float> min_size = min_sizes[i];
        // cout << feature_map_size.height << " " << feature_map_size.width << endl;

        for (auto _h = 0; _h < feature_map_size.height; ++_h) {
            for (auto _w = 0; _w < feature_map_size.width; ++_w) {
                for (auto j = 0; j < min_size.size(); ++j) {
                    float s_kx = min_size[j] / in_w;
                    float s_ky = min_size[j] / in_h;

                    float cx = (_w + 0.5) * steps[i] / in_w;
                    float cy = (_h + 0.5) * steps[i] / in_h;

                    // if (i == 3) {
                    //     cout << _h << " " << _w << endl;
                    // }

                    BndBox_xywh anchor = { {cx, cy}, s_kx, s_ky };
                    anchors.push_back(anchor);
                }
            }
        }
    }
    // cout << "Out!" << endl;
    return anchors;
}

vector<Face> PriorBox::decode(const Mat& loc,
    const Mat& conf) {
    vector<Face> dets; // num * [x1, y1, x2, y2, x_re, y_re, x_le, y_le, x_ml, y_ml, x_n, y_n, x_mr, y_ml]

    float* loc_v = (float*)(loc.data);
    float* conf_v = (float*)(conf.data);
    for (auto i = 0; i < priors.size(); ++i) {
        float cx = priors[i].center.x + loc_v[i * 14 + 0] * variance[0] * priors[i].w;
        float cy = priors[i].center.y + loc_v[i * 14 + 1] * variance[0] * priors[i].h;
        float w = priors[i].w * exp(loc_v[i * 14 + 2] * variance[0]);
        float h = priors[i].h * exp(loc_v[i * 14 + 3] * variance[1]);

        // get bounding box
        float x1 = (cx - w / 2) * out_w;
        float y1 = (cy - h / 2) * out_h;
        float x2 = (cx + w / 2) * out_w;
        float y2 = (cy + h / 2) * out_h;
        BndBox_xyxy bbox = { {x1, y1}, {x2, y2} };
        // get landmarks, loc->[right_eye, left_eye, mouth_left, nose, mouth_right]
        float x_re = (priors[i].center.x + loc_v[i * 14 + 4] * variance[0] * priors[i].w) * out_w;
        float y_re = (priors[i].center.y + loc_v[i * 14 + 5] * variance[0] * priors[i].h) * out_h;
        float x_le = (priors[i].center.x + loc_v[i * 14 + 6] * variance[0] * priors[i].w) * out_w;
        float y_le = (priors[i].center.y + loc_v[i * 14 + 7] * variance[0] * priors[i].h) * out_h;
        float x_ml = (priors[i].center.x + loc_v[i * 14 + 8] * variance[0] * priors[i].w) * out_w;
        float y_ml = (priors[i].center.y + loc_v[i * 14 + 9] * variance[0] * priors[i].h) * out_h;
        float x_n = (priors[i].center.x + loc_v[i * 14 + 10] * variance[0] * priors[i].w) * out_w;
        float y_n = (priors[i].center.y + loc_v[i * 14 + 11] * variance[0] * priors[i].h) * out_h;
        float x_mr = (priors[i].center.x + loc_v[i * 14 + 12] * variance[0] * priors[i].w) * out_w;
        float y_mr = (priors[i].center.y + loc_v[i * 14 + 13] * variance[0] * priors[i].h) * out_h;
        Landmarks_10 landmarks = { {x_re, y_re}, // right eye
                                   {x_le, y_le}, // left eye
                                   {x_ml, y_ml}, // mouth left
                                   {x_n,  y_n },  // nose
                                   {x_mr, y_mr}  // mouth right
        };
        // get score
        float score = conf_v[i * 2 + 1];


        Face det = { bbox, landmarks, score };
        dets.push_back(det);
    }

    return dets;
}

Size libface::get_input_shape(string model_fpath)
{
    size_t start = model_fpath.find("_") + 1;
    size_t end = model_fpath.find(".onnx");
    int width = stoi(model_fpath.substr(start, end - start));
    int height = int(0.75 * width);

    return Size(width, height);
}

// dets is of dimension [num, 15], which is 
// num * [x1, y1, x2, y2, x_re, y_re, x_le, y_le, x_ml, y_ml, x_n, y_n, x_mr, y_ml, label]
void libface::nms(vector<Face>& dets, const float thresh) 
{
    sort(dets.begin(), dets.end(), [](const Face& a, const Face& b) { return a.score > b.score; });

    // vector<Face> post_nms;
    vector<bool> isSuppressed(dets.size(), false);
    for (auto i = 0; i < dets.size(); ++i) {
        if (isSuppressed[i]) { continue; }

        // area of i bbox
        float area_i = dets[i].bbox.area();
        for (auto j = i + 1; j < dets.size(); ++j) {
            if (isSuppressed[j]) { continue; }

            // area of intersection
            float ix1 = max(dets[i].bbox.top_left.x, dets[j].bbox.top_left.x);
            float iy1 = max(dets[i].bbox.top_left.y, dets[j].bbox.top_left.y);
            float ix2 = min(dets[i].bbox.bottom_right.x, dets[j].bbox.bottom_right.x);
            float iy2 = min(dets[i].bbox.bottom_right.y, dets[j].bbox.bottom_right.y);

            float iw = ix2 - ix1 + 1;
            float ih = iy2 - iy1 + 1;
            if (iw <= 0 || ih <= 0) { continue; }
            float inter = iw * ih;

            // area of j bbox
            float area_j = dets[j].bbox.area();

            // iou
            float iou = inter / (area_i + area_j - inter);
            if (iou > thresh) { isSuppressed[j] = true; }
        }
        // post_nms.push_back(dets[i]);
    }
    // return post_nms;
    int idx_t = 0;
    dets.erase(
        remove_if(dets.begin(), dets.end(), [&idx_t, &isSuppressed](const Face& f) { return isSuppressed[idx_t++]; }),
        dets.end()
    );
}

vector<Face> libface::detect(Mat img)
{
    Size output_shape = img.size();
    Mat img_resize;
    resize(img, img_resize, input_shape);
    Mat blob = blobFromImage(img_resize, 1.0, input_shape);

    // Forward
    //vector<String> output_names = { "loc", "conf" };
    vector<Mat> output_blobs;
    this->net.setInput(blob);
    this->net.forward(output_blobs, this->output_names);

    // Decode bboxes, landmarks and scores
    PriorBox pb(this->input_shape, output_shape);
    vector<Face> dets = pb.decode(output_blobs[0], output_blobs[1]);

    // Ignore low scores
    const float conf_thresh_ = this->conf_thresh;
    dets.erase(remove_if(dets.begin(), dets.end(), [&conf_thresh_](const Face& f) { return f.score <= conf_thresh_; }), dets.end());
    
    // NMS
    if (dets.size() > 1) 
    {
        this->nms(dets, this->nms_thresh);
        if (dets.size() > this->keep_top_k) 
        { 
            dets.erase(dets.begin() + this->keep_top_k, dets.end()); 
        }
    }
    /*else if (dets.size() < 1) 
    {
        cout << "No faces found." << endl;
        return dets;
    }*/
    /*cout << "Detection results: " << dets.size() << " faces found." << endl;
    for (auto i = 0; i < dets.size(); ++i) 
    {
        BndBox_xyxy bbox = dets[i].bbox;
        float score = dets[i].score;
        cout << bbox.top_left << " " << bbox.bottom_right << " " << score << endl;
    }*/
    return dets;
}

Mat libface::crop_face(Face det, Mat srcimg)
{
    Mat face_roi;
    if (this->align)
    {    ////如果简单地计算左眼和右眼连成直线的倾斜角,然后对做旋转,得到的图像的边界有黑色
        const int image_size[2] = {224, 224};
        //float REFERENCE_FACIAL_POINTS[5*2] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252, 71.7366, 33.5493, 92.3655, 62.7299, 92.2041};   //REFERENCE_FACIAL_POINTS[:, 0] += 8 , REFERENCE_FACIAL_POINTS[:, 1] -= 8
        float REFERENCE_FACIAL_POINTS[5 * 2] = {38.2946, 43.6963, 73.5318, 43.5014, 56.0252, 63.7366, 41.5493, 84.3655, 70.7299, 84.2041};  //你也可以把这个数组写作类的私有成员变量
        int i = 0;
        float x = 0, y = 0;
        vector<Point2f> dst, src;
        if (image_size[0] == image_size[1] and image_size[0] != 112)
        {
            for (i = 0; i < 5; i++)
            {
                x = REFERENCE_FACIAL_POINTS[i * 2] / 112 * image_size[0];
                y = REFERENCE_FACIAL_POINTS[i * 2 + 1] / 112 * image_size[0];
                src.push_back(Point2f(x, y));
            }
        }
        ///landmarks里的存储顺序是：[right_eye, left_eye, mouth_left, nose, mouth_right]
        dst.push_back(det.landmarks.right_eye);
        dst.push_back(det.landmarks.left_eye);
        dst.push_back(det.landmarks.mouth_left);
        dst.push_back(det.landmarks.nose_tip);
        dst.push_back(det.landmarks.mouth_right);
        vector<uchar> inliers(dst.size(), 0);
        Mat M = estimateAffinePartial2D(dst, src, inliers);
        Size outSize(image_size[1], image_size[0]);
        warpAffine(srcimg, face_roi, M, outSize);
    }
    else
    {
        int x_start = MaxInt(0, det.bbox.top_left.x);
        int y_start = MaxInt(0, det.bbox.top_left.y);
        int x_end = MinInt(srcimg.cols, det.bbox.bottom_right.x);
        int y_end = MinInt(srcimg.rows, det.bbox.bottom_right.y);
        Rect rect(x_start, y_start, x_end - x_start, y_end - y_start);  ///防止超越图像边界
        face_roi = srcimg(rect);
    }
    return face_roi;
}

void libface::draw(Mat& img, const vector<Face>& faces, bool is_draw_lanmarks)
{

    const int thickness = 2;
    const Scalar bbox_color = { 0, 0,   255 };
    const Scalar text_color = { 255, 255, 255 };
    const vector<Scalar> landmarks_color = {
        {255,   0,   0}, // left eye
        {  0,   0, 255}, // right eye
        {  0, 255, 255}, // mouth left
        {255, 255,   0}, // nose
        {  0, 255,   0}  // mouth right
    };

    auto point2f2point = [](Point2f p, bool shift = false) {
        return shift ? Point(int(p.x), int(p.y) + 12) : Point(int(p.x), int(p.y));
    };
    for (auto i = 0; i < faces.size(); ++i)
    {
        // draw bbox
        rectangle(img,
            point2f2point(faces[i].bbox.top_left),
            point2f2point(faces[i].bbox.bottom_right),
            bbox_color,
            thickness);
        // put score by the corner of bbox
        string str_score = to_string(faces[i].score);
        if (str_score.size() > 6) {
            str_score.erase(6);
        }
        putText(img,
            str_score,
            point2f2point(faces[i].bbox.top_left, true),
            FONT_HERSHEY_DUPLEX,
            0.5, // Font scale
            text_color);
        // draw landmarks
        if (is_draw_lanmarks)
        {
            const int radius = 2;
            circle(img, point2f2point(faces[i].landmarks.left_eye), radius, landmarks_color[0], thickness);
            circle(img, point2f2point(faces[i].landmarks.right_eye), radius, landmarks_color[1], thickness);
            circle(img, point2f2point(faces[i].landmarks.mouth_left), radius, landmarks_color[2], thickness);
            circle(img, point2f2point(faces[i].landmarks.nose_tip), radius, landmarks_color[3], thickness);
            circle(img, point2f2point(faces[i].landmarks.mouth_right), radius, landmarks_color[4], thickness);
        } 
    }
}

vector<float> openface::get_feature(Mat img)
{
    Mat blob = blobFromImage(img, 1.0 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    Mat output_blobs = this->net.forward();

    vector<float> output(this->length, 0);
    for (int i = 0; i < this->length; i++)
    {
        output[i] = (float)output_blobs.data[i];
        //cout << output[i] << endl;
    }
    //Min_Max_Normalization(output);
    Unit_Normalization(output);
    return output;
}

vector<float> arcface::get_feature(Mat img)
{
    Mat img_resize;
    cvtColor(img, img_resize, COLOR_BGR2GRAY);
    resize(img_resize, img_resize, Size(this->inpWidth, this->inpHeight), INTER_AREA);

    Mat blob;
    blobFromImage(img_resize, blob, 1 / 127.5, Size(), Scalar::all(127.5));
    this->net.setInput(blob);

    Mat output_blobs = this->net.forward();
    vector<float> output(this->length, 0);
    for (int i = 0; i < this->length; i++)
    {
        output[i] = (float)output_blobs.data[i];
    }
    //Min_Max_Normalization(output);
    Unit_Normalization(output);
    return output;
}

vector<Point> pfld::detect(Mat crop_img)
{
    Mat blob = blobFromImage(crop_img, 1.0 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true);
    this->net.setInput(blob);
    vector<Mat> output_blobs;
    this->net.forward(output_blobs, this->output_names);
    float* landmarks = (float*)(output_blobs[1].data);
    vector<Point> pre_landmark;
    int num_points = (int)(output_blobs[1].cols * 0.5);
    int x = 0, y = 0;
    for (int i = 0; i < num_points; i++)
    {
        x = (int)(landmarks[i * 2] * crop_img.cols);
        y = (int)(landmarks[i * 2 + 1] * crop_img.rows);
        pre_landmark.push_back(Point(x,y));
    }
    return pre_landmark;
}

void pfld::draw_landmarks(vector<Point> pts, Mat& img)
{
    for (int i = 0; i < pts.size(); i++)
    {
        circle(img, pts[i], 2, Scalar(0, 255, 0), -1);
    }
}

void pfld::face_detect_draw_landmarks(vector<Point> pts, Mat& img, int start_x, int start_y)
{
    for (int i = 0; i < pts.size(); i++)
    {
        circle(img, Point(pts[i].x + start_x, pts[i].y + start_y), 2, Scalar(0, 255, 0), -1);
    }
}
