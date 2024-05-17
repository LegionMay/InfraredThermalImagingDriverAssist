// Part 1: 包含必要的头文件和命名空间
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/c/common.h>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

// Part 2: 预处理函数，调整图像尺寸和数据类型
void preprocess(Mat& src, Mat& dst, int input_width, int input_height) {
    resize(src, dst, Size(input_width, input_height));
}

// Part 3: 设置输入张量
void setInputTensor(unique_ptr<tflite::Interpreter>& interpreter, Mat& input_image) {
    int input_tensor_size = interpreter->input_tensor(0)->bytes;
    TfLiteIntArray* dims = interpreter->input_tensor(0)->dims;
    int desired_height = dims->data[1];
    int desired_width = dims->data[2];
    int desired_channels = dims->data[3];

    preprocess(input_image, input_image, desired_width, desired_height);

    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);

    if (input_image.total() * input_image.elemSize() != input_tensor_size) {
        cerr << "预处理后的图像大小与输入张量大小不匹配！" << endl;
        return;
    }

    memcpy(input, input_image.data, input_tensor_size);
}

// Part 4: 获取检测结果
vector<tuple<string, float, Rect>> getDetections(unique_ptr<tflite::Interpreter>& interpreter, const vector<string>& labels, float threshold = 0.5) {
    const float* output_locations = interpreter->typed_output_tensor<float>(0);  // Bounding boxes
    const float* output_classes = interpreter->typed_output_tensor<float>(1);    // Class IDs
    const float* output_scores = interpreter->typed_output_tensor<float>(2);     // Scores
    const int num_detections = static_cast<int>(interpreter->typed_output_tensor<float>(3)[0]);

    vector<tuple<string, float, Rect>> detections;
    for (int i = 0; i < num_detections; ++i) {
        float score = output_scores[i];
        if (score < threshold) continue;

        int class_id = static_cast<int>(output_classes[i]);
        string label = labels[class_id];
        float ymin = output_locations[4 * i] * 640;  // Scale according to image size
        float xmin = output_locations[4 * i + 1] * 480;
        float ymax = output_locations[4 * i + 2] * 640;
        float xmax = output_locations[4 * i + 3] * 480;
        Rect rect(Point(xmin, ymin), Point(xmax, ymax));
        detections.emplace_back(label, score, rect);
    }

    return detections;
}

// Part 5: 主函数
int main() {
    VideoCapture cap(CAP_V4L2);

    if (!cap.isOpened()) {
        cerr << "摄像头打开失败！" << endl;
        return -1;
    }

    // 设置摄像头参数
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    Mat frame, resized_frame;

    namedWindow("Processed Frame", WINDOW_NORMAL);
    setWindowProperty("Processed Frame", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    // 加载模型
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("/usr/local/demo-ai/computer-vision/models/coco_ssd_mobilenet/detect.tflite");

    if (!model) {
        cerr << "模型加载失败！" << endl;
        return -1;
    }

    // 创建解释器
    tflite::ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        cerr << "解释器创建失败！" << endl;
        return -1;
    }

    interpreter->SetNumThreads(1);

    // 分配张量内存
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "张量内存分配失败！" << endl;
        return -1;
    }

    // 获取输入张量的尺寸
    int input_width = interpreter->input_tensor(0)->dims->data[1];
    int input_height = interpreter->input_tensor(0)->dims->data[2];

    // 加载标签
    vector<string> labels;
    ifstream labelsFile("/usr/local/demo-ai/computer-vision/models/coco_ssd_mobilenet/labels.txt");
    string line;
    while (getline(labelsFile, line)) {
        labels.push_back(line);
    }

    // 处理帧
    auto start_time = chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "捕获帧失败！" << endl;
            break;
        }

        // 旋转和预处理图像
        rotate(frame, frame, ROTATE_90_CLOCKWISE);
        preprocess(frame, resized_frame, input_width, input_height);
        setInputTensor(interpreter, resized_frame);

        // 进行推理
        if (interpreter->Invoke() != kTfLiteOk) {
            cerr << "推理失败！" << endl;
            return -1;
        }

        // 获取检测结果并绘制
        vector<tuple<string, float, Rect>> detections = getDetections(interpreter, labels);
        for (const auto& detection : detections) {
            const string& label = get<0>(detection);
            float score = get<1>(detection);
            const Rect& rect = get<2>(detection);
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
            putText(frame, label + ": " + to_string(score), rect.tl(), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        }

        // 显示处理后的图像
        imshow("Processed Frame", frame);

        // 按 'q' 退出
        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
