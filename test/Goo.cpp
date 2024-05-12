
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>

int main() {
    // 使用 V4L2 作为 OpenCV 的视频捕获后端
    cv::VideoCapture cap(cv::CAP_V4L2, 0);

    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
    std::cerr << "摄像头打开失败！" << std::endl;
    return -1;
    }

    // 设置捕获参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    cv::Mat frame, resized_frame, rotated_frame;

    // 创建一个全屏窗口
    cv::namedWindow("Rotated Frame", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Rotated Frame", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    while (true) {
        // 从摄像头捕获一帧
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "帧捕获失败！" << std::endl;
            break;
            }

        // 使用 OpenCV 的 resize 函数来放大图像
        cv::resize(frame, resized_frame, cv::Size(480, 640));

        // 顺时针旋转90度
        cv::rotate(resized_frame, rotated_frame, cv::ROTATE_90_CLOCKWISE);
        
        // 显示旋转后的图像
        cv::imshow("Rotated Frame", rotated_frame);
        
    }
}
