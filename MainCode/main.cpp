#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <edgetpu.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstdlib>

#define WINDOW_NAME "Video Window"
#define SETTINGS_WINDOW "Settings"
#define FRAME_SKIP 4
#define FRAME_HOLD 1
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 480

using namespace std;
using namespace cv;


bool alarm_on = false; // 报警开关
bool tcp_start = false; 
bool tcp_success = false;
bool display_video0 = false;
bool button_pressed = false;
bool settings_page = false;
bool running = true;
bool capture_done = false;
bool mode = true;  // true为热成像模式, false为热融合模式

int slider1_value = 0, slider2_value = 0;
int frame_skip_count = FRAME_SKIP;
int frame_hold_count = FRAME_HOLD;
int client_sock = -1;
float detection_threshold = 0.5f;
// 按钮大小和位置调整
Rect switch_button_rect(650, 20, 130, 50);
Rect settings_button_rect(650, 210, 130, 50);
Rect exit_button_rect(650, 420, 130, 50);
Rect back_button_rect(650, 20, 130, 50);
Rect slider1_rect(50, 100, 600, 50);
Rect slider2_rect(50, 200, 600, 50);
Rect toggle1_rect(50, 300, 300, 75);
Rect toggle2_rect(400, 300, 300, 75);
Rect buttonRect1(640, 0, 160, 160); 
Rect buttonRect2(640, 160, 160, 160);
Rect buttonRect3(640, 320, 160, 160);

Mat frame0, frame2, processed_frame;
Mat settings_background;
chrono::time_point<chrono::steady_clock> last_time;
int frame_count = 0;
vector<string> labels;
vector<vector<tuple<string, float, Rect>>> detection_history;

unique_ptr<tflite::FlatBufferModel> model;
shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
unique_ptr<tflite::Interpreter> interpreter;

void playSound() {
    system("aplay -D hw:0,0 /home/root/usr/MP157/waralarm.wav &");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    system("killall aplay");
}

// 预处理函数
void preprocess(Mat& src, Mat& dst, int input_width, int input_height) {
    resize(src, dst, Size(input_width, input_height));
}

// 设置输入张量
void setInputTensor(tflite::Interpreter* interpreter, Mat& input_image) {
    uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
    if (!input) {
        cerr << "获取输入张量失败!" << endl;
        return;
    }
    memcpy(input, input_image.data, input_image.total() * input_image.elemSize());
}

// 获取检测结果
vector<tuple<string, float, Rect>> getDetections(tflite::Interpreter* interpreter, const vector<string>& labels, float detection_threshold) {
    const float* output_locations = interpreter->typed_output_tensor<float>(0);
    const float* output_classes = interpreter->typed_output_tensor<float>(1);
    const float* output_scores = interpreter->typed_output_tensor<float>(2);
    int num_detections = static_cast<int>(interpreter->typed_output_tensor<float>(3)[0]);

    vector<tuple<string, float, Rect>> detections;
    for (int i = 0; i < num_detections; ++i) {
        float score = output_scores[i];
        if (score < detection_threshold) continue;

        int class_id = static_cast<int>(output_classes[i]);
        string label = labels[class_id];
        float ymin = output_locations[4 * i] * interpreter->input_tensor(0)->dims->data[1];
        float xmin = output_locations[4 * i + 1] * interpreter->input_tensor(0)->dims->data[2];
        float ymax = output_locations[4 * i + 2] * interpreter->input_tensor(0)->dims->data[1];
        float xmax = output_locations[4 * i + 3] * interpreter->input_tensor(0)->dims->data[2];
        Rect rect(Point(static_cast<int>(xmin), static_cast<int>(ymin)), 
                  Point(static_cast<int>(xmax), static_cast<int>(ymax)));
        detections.emplace_back(label, score, rect);
    }

    return detections;
}

// 在图像上绘制检测结果
void drawDetections(Mat& frame, const vector<tuple<string, float, Rect>>& detections) {
    for (const auto& detection : detections) {
        const string& label = get<0>(detection);
        float score = get<1>(detection);
        const Rect& rect = get<2>(detection);
        rectangle(frame, rect, Scalar(0, 0, 255), 2);
        
    }
}

// 格式化检测信息并发送
void sendDetections(const vector<tuple<string, float, Rect>>& detections) {
    stringstream ss;
    ss << "Detections: " << detections.size() << "\n";
    for (const auto& detection : detections) {
        const Rect& rect = get<2>(detection);
        ss << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << "\n";
    }
    if (tcp_success) {
        string detection_info = ss.str();
        send(client_sock, detection_info.c_str(), detection_info.size(), 0);
    }
}

// 尝试建立TCP连接
void tryTcpConnection() {
    if (client_sock != -1) {
        close(client_sock);
    }

    client_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock == -1) {
        cerr << "创建套接字失败!" << endl;
        tcp_success = false;
        return;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(5555);
    server_addr.sin_addr.s_addr = inet_addr("192.168.5.8");

    if (connect(client_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        cerr << "连接服务器失败！" << endl;
        close(client_sock);
        client_sock = -1;
        tcp_success = false;
        return;
    }

    tcp_success = true;
}

// 鼠标点击回调函数
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        if (!settings_page) {
            if (switch_button_rect.contains(Point(x, y))) {
                button_pressed = true;
            } else if (settings_button_rect.contains(Point(x, y))) {
                settings_page = true;
            } else if (exit_button_rect.contains(Point(x, y))) {
                running = false;
            }
        } else {
            if (back_button_rect.contains(Point(x, y))) {
                settings_page = false;
            } else if (slider1_rect.contains(Point(x, y))) {
                slider1_value = (x - slider1_rect.x) * 100 / slider1_rect.width;
                detection_threshold = slider1_value / 100.0f;
            } else if (slider2_rect.contains(Point(x, y))) {
                slider2_value = (x - slider2_rect.x) * 100 / slider2_rect.width;
            }
            if (toggle1_rect.contains(Point(x, y))) {
                alarm_on = !alarm_on; // 切换报警开关状态
            }
            if (toggle2_rect.contains(Point(x, y))) {
                tcp_start = !tcp_start; // 切换开关状态
                if (tcp_start) {
                    tryTcpConnection();
                } else {
                    if (client_sock != -1) {
                        close(client_sock);
                        client_sock = -1;
                    }
                    tcp_success = false;
                }
            }
        }
    } else if (event == EVENT_LBUTTONUP) {
        if (!settings_page && switch_button_rect.contains(Point(x, y)) && button_pressed) {
            display_video0 = !display_video0;
            button_pressed = false;
        }
    }
}

// 绘制按钮
void drawButton(Mat& frame, const Rect& rect, const string& text, Scalar color = Scalar(100, 100, 100), int thickness = 2) {
    rectangle(frame, rect, color, FILLED, LINE_AA);
    rectangle(frame, rect, Scalar(255, 255, 255), thickness, LINE_AA);
    putText(frame, text, rect.tl() + Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
}

// 绘制滑块
void drawSlider(Mat& frame, const Rect& rect, const string& text, int value, Scalar color = Scalar(100, 100, 100), int thickness = 2) {
    rectangle(frame, rect, color, FILLED, LINE_AA);
    rectangle(frame, rect, Scalar(255, 255, 255), thickness, LINE_AA);
    putText(frame, text, rect.tl() + Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    int slider_position = rect.x + value * rect.width / 100;
    line(frame, Point(slider_position, rect.y), Point(slider_position, rect.y + rect.height), Scalar(0, 0, 255), 2, LINE_AA);
}

// 绘制开关
void drawToggle(Mat& frame, const Rect& rect, const string& text, Scalar color = Scalar(100, 100, 100), int thickness = 2) {
    rectangle(frame, rect, color, FILLED, LINE_AA);
    rectangle(frame, rect, Scalar(255, 255, 255), thickness, LINE_AA);
    putText(frame, text, rect.tl() + Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
}

// 主函数
int main() {
    // 打开热成像摄像头
    VideoCapture cap0("/dev/video2", CAP_V4L2);
    if (!cap0.isOpened()) {
        cerr << "打开视频流0失败!" << endl;
        return -1;
    }
    cap0.set(CAP_PROP_FRAME_WIDTH, 320);
    cap0.set(CAP_PROP_FRAME_HEIGHT, 240);
    cap0.set(CAP_PROP_FPS, 30);
    cap0.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // 打开可见光摄像头
    VideoCapture cap2("/dev/video0", CAP_V4L2);
    if (!cap2.isOpened()) {
        cerr << "打开视频流2失败!" << endl;
        return -1;
    }
    cap2.set(CAP_PROP_FRAME_WIDTH, 320);
    cap2.set(CAP_PROP_FRAME_HEIGHT, 240);
    cap2.set(CAP_PROP_FPS, 30);
    cap2.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // 加载模型
    model = tflite::FlatBufferModel::BuildFromFile("/home/root/usr/NewTPU/Models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite");
    if (!model) {
        cerr << "加载模型失败!" << endl;
        return -1;
    }

    // 创建Edge TPU上下文
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(edgetpu::DeviceType::kApexUsb);
    if (!edgetpu_context) {
        cerr << "打开Edge TPU设备失败!" << endl;
        return -1;
    }

    // 创建解释器
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        cerr << "构建解释器失败!" << endl;
        return -1;
    }
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(4);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "分配张量失败!" << endl;
        return -1;
    }

    // 加载标签
    ifstream label_file("/usr/local/demo-ai/computer-vision/models/coco_ssd_mobilenet/labels.txt");
    string line;
    while (getline(label_file, line)) {
        labels.push_back(line);
    }

    // 创建窗口并设置鼠标回调
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    setMouseCallback(WINDOW_NAME, onMouse);
    resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT);
    setWindowProperty(WINDOW_NAME, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    // 初始化变量
    auto start = chrono::steady_clock::now();
    int frame_counter = 0;
    settings_background = imread("/home/root/usr/MP157/logo.png");
    if (settings_background.empty()) {
        cerr << "加载背景图片失败!" << endl;
        return -1;
    }
    slider1_value = 30;
    slider2_value = 50;
    bool alarm_condition_met = false;

    while (running) {
        Mat display_frame(480, 800, CV_8UC3);

        cap0 >> frame0;
        if (frame0.empty()) {
            cerr << "捕获帧失败！" << endl;
            break;
        }

        rotate(frame0, frame0, ROTATE_90_CLOCKWISE);
        Mat resized_frame;
        preprocess(frame0, resized_frame, interpreter->input_tensor(0)->dims->data[2], interpreter->input_tensor(0)->dims->data[1]);
        setInputTensor(interpreter.get(), resized_frame);

        if (display_video0) {
            if (frame_skip_count == 0) {
                if (interpreter->Invoke() != kTfLiteOk) {
                    cerr << "推理失败！" << endl;
                    continue;
                }

                vector<tuple<string, float, Rect>> detections = getDetections(interpreter.get(), labels, detection_threshold);
                processed_frame = frame0.clone();
                drawDetections(processed_frame, detections);

                detection_history.push_back(detections);
                if (detection_history.size() > frame_hold_count) {
                    detection_history.erase(detection_history.begin());
                }

                bool alarm_condition_met = false;
                for (const auto& detection : detections) {
                    const Rect& rect = get<2>(detection);
                    if (rect.area() > slider2_value * 800) {
                        alarm_condition_met = true;
                        break;
                    }
                }
                if (alarm_on && alarm_condition_met) {
                    std::thread soundThread(playSound);
                    soundThread.detach(); // 分离线程，使其在后台运行
                }
                // 发送检测信息
                sendDetections(detections);
                frame_skip_count = FRAME_SKIP;
            } else {
                frame_skip_count--;
                processed_frame = frame0.clone();

                for (const auto& detections : detection_history) {
                    drawDetections(processed_frame, detections);
                }

                // 发送历史检测信息
                if (!detection_history.empty()) {
                    sendDetections(detection_history.back());
                }
            }
        } else {
            cap2 >> frame2;
            if (!frame2.empty() && !frame0.empty()) {
                rotate(frame2, frame2, ROTATE_90_CLOCKWISE);
                processed_frame = frame2.clone();
                Mat frame0_processed = frame0.clone();
                for (int i = 0; i < frame0_processed.rows; ++i) {
                    for (int j = 0; j < frame0_processed.cols; ++j) {
                        Vec3b& pixel = frame0_processed.at<Vec3b>(i, j);
                        if (pixel[0] > 240 && pixel[1] > 240 && pixel[2] > 240) {  // 白色部分
                            pixel = Vec3b(0, 255, 255);  // 变为黄色
                        }
                    }
                }

                // 定义裁剪区域 (x, y, width, height)
                Rect crop_region(40, 45, 240, 180);
                // 裁剪图像
                frame0_processed = frame0_processed(crop_region);

                // 确保 frame0_processed 和 frame2 尺寸一致
                if (frame0_processed.size() != frame2.size()) {
                    resize(frame0_processed, frame0_processed, frame2.size());
                }

                if (!frame2.empty()) {
                    addWeighted(frame2, 0.8, frame0_processed, 0.2, 0, frame2);
                    processed_frame = frame2.clone();
                }

                for (const auto& detections : detection_history) {
                    drawDetections(frame2, detections);
                }
            }
        }

        if (settings_page) {
            string slider1_value_str = to_string(slider1_value / 100.0f);
            string slider2_value_str = to_string(slider2_value / 100.0f);
            slider1_value_str = slider1_value_str.substr(0, slider1_value_str.find(".") + 3);
            slider2_value_str = slider2_value_str.substr(0, slider2_value_str.find(".") + 3);
            putText(display_frame, slider1_value_str, Point(650, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            putText(display_frame, slider2_value_str, Point(650, 250), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            putText(display_frame, "Geek_Republic   10440", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
            drawButton(display_frame, back_button_rect, "Back");
            drawSlider(display_frame, slider1_rect, "DetecThreshold", slider1_value);
            drawSlider(display_frame, slider2_rect, "AlarmThreshold", slider2_value);
            drawToggle(display_frame, toggle1_rect, alarm_on ? "Alarm(ON)" : "Alarm(OFF)");
            drawToggle(display_frame, toggle2_rect, tcp_start ? "TCP(START)" : "TCP(OFF)");
            if (tcp_start)
                putText(display_frame, tcp_success ? "TCP Successed" : "TCP Failed", Point(400, 450), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
            if (display_frame.size() != settings_background.size()) {
                resize(settings_background, settings_background, display_frame.size());
            }
            addWeighted(display_frame, 0.6, settings_background, 0.4, 0, display_frame);
        } else {
            Mat processed_frame_copy;
            processed_frame_copy = processed_frame.clone();
            if (!processed_frame_copy.empty()) {
                Mat roi = display_frame(Rect(0, 0, 640, 480));
                resize(processed_frame_copy, roi, roi.size());
            }
            drawButton(display_frame, switch_button_rect, display_video0 ? "MODE1" : "MODE2");
            drawButton(display_frame, settings_button_rect, "SET");
            drawButton(display_frame, exit_button_rect, "EXIT");
        }

        imshow(WINDOW_NAME, display_frame);
        if (waitKey(10) == 27) {
            running = false;
        }
    }

    // 关闭套接字
    if (client_sock != -1) {
        close(client_sock);
    }

    return 0;
}

