#include <opencv2/opencv.hpp>
#include <iostream>
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
cap.set(cv::CAP_PROP_FPS, 25);
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

// 创建管道并启动 MPV 播放器
FILE* pipe = popen("ffmpeg -y -f mjpeg -vcodec mjpeg -i - -vf \"scale=640:480,transpose=1\" -b:v 8000k -an -f avi pipe:1 | mpv -", "w");

if (!pipe) {
std::cerr << "管道创建失败！" << std::endl;
return -1;
}

cv::Mat frame;
std::vector<unsigned char> buffer;
while (true) {
// 从摄像头捕获一帧
cap >> frame;
if (frame.empty()) {
std::cerr << "帧捕获失败！" << std::endl;
break;
}

// 使用 imencode 函数将帧编码为 JPEG 格式
if (!cv::imencode(".jpg", frame, buffer)) {
std::cerr << "帧编码失败！" << std::endl;
break;
}

// 将编码后的帧写入管道
fwrite(buffer.data(), 1, buffer.size(), pipe);

// 检查用户是否按下 'q' 键退出
if (cv::waitKey(1) == 'q') {
break;
}
}

// 关闭管道和摄像头
pclose(pipe);
cap.release();

return 0;
}
