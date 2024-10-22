# InfraredThermalImagingDriverAssist  
部分效果演示视频：[STM32MP157热成像融合与目标检测简单应用](https://www.bilibili.com/video/BV1zry3YSEwY/?vd_source=65dfce85ac7d56ef11d4a2db99106b8c)
## 前言  
随着汽车行业的快速发展，驾驶安全已成为公众关注的焦点。据交通安全部门统计，低能见度环境下的交通事故率明显高于正常天气条件。传统的驾驶辅助系统依赖于可见光摄像头，但在雾天、夜间等低能见度环境下，往往难以准确识别道路障碍物，增加了交通事故的风险。为了解决这一问题，我们开发了一款基于STM32MP157F-DK2的红外热融合智能车载驾驶辅助系统，通过在该主控平台上部署Linux操作系统，结合可见光/近红外摄像头和红外热成像模块，并采用图像处理算法和嵌入式人工智能技术，实现在低能见度环境下对障碍物、人员、车辆的智能识别和碰撞预警，从而提高驾驶安全性。我们期望该系统能够有效解决低能见度环境下的驾驶安全问题，减少因视线不良导致的交通事故，提升夜间或恶劣天气条件下的驾驶体验，同时为智能驾驶辅助系统的发展带来新的机遇。  
## 1. 整装待发
### 1.1 配置开发环境，点亮LCD  
这里选择直接在Ubuntu上进行开发，为方便文件的传输，使用```sudo vmhgfs-fuse .host:/<共享文件夹名> /mnt/hgfs -o subtype=vmhgfs-fuse,allow_other```命令将一个共享文件夹挂载在Ubuntu的/mnt/hgfs目录。  
开发环境的配置通常是一个繁琐的过程，这里参考了ST官方的教程 [STM32MP157x-DK2](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/Getting_started/STM32MP1_boards/STM32MP157x-DK2/Develop_on_Arm%C2%AE_Cortex%C2%AE-A7 )，一步一步跟着来。 
同时直接使用了参加st线下buidroot活动时获得的其官方配置好的虚拟机，Ubuntu版本为22.04。  
值得注意的是使用CubeProgrammer烧录时自动生成的二进制文件存储路径可能不准确，而且手动配置时会自动添加后缀。另外STM32MP157F-DK2实测供电至少为5V 2A才能点亮屏幕。
注意：需要手动设置开发板系统时间为最新，否则```apt-get update```可能报错以致无法安装某些软件包。  
烧录好STM32MP1 OpenSTLinux Starter Package并安装好SDK后，我们就可以开始编写第一个程序啦！  

### 1.2 为开发板编写第一个Hello World程序  
仍然参考上文所述的ST官方教程，同时参考了这篇CSDN笔记[STM32MP157开发笔记 ](https://mculover666.blog.csdn.net/article/details/121952359?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-121952359-blog-121969843.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-121952359-blog-121969843.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=5) 使用```minicom -D /dev/ttyACM0```命令来连接开发板串口进行操作。  
开发板与PC需要先联网，如果使用usb网卡注意连接主机而非虚拟机。由于这一平台只有一个网口，每次连接主机后需要设置开发板IP如 ```ifconfig end0 192.168.5.9```,而连接到外网后又要重新配置IP如```udhcpc -i end0``` 否则无法正常建立连接。     
我们按照教程创建并编写一个.c文件及其makefile文件，编译完成后通过这段命令 ```scp gtk_hello_world root@<board ip address>:~/usr/local``` 将可执行文件推到开发板上，然后通过串口操作开发板运行这一程序，成功在开发板上执行了这个Hello World程序。  

### 1.3 在Ubuntu上接入热成像模块进行测试  
这里用以下命令实现了在Ubuntu上实时采集热成像画面：  
```
ffmpeg -f v4l2 -s 240x320 -r 25 -vcodec mjpeg -i /dev/video1 -b:v 8000k -an -f avi - | ffplay -f avi -
``` 
<img width="1280" alt="b8eccb8ee96b8c529203ed0382b348e" src="https://github.com/LegionMay/InfraredThermalImagingDriverAssist/assets/110379545/faec65d4-2e47-4bcd-a5f9-2a2573ada848">  

## 2 初具雏形
### 2.1 为开发板安装X-LINUX-AI软件包并测试热成像模块    
参考[X-LINUX-AI 入门包](https://wiki.st.com/stm32mpu/wiki/X-LINUX-AI_Starter_package)安装X-LINUX-AI软件包，其中包括了我们所需的opencv和tensorflow lite的动态链接库。  
通过这段命令拍摄一段热成像画面进行测试（需要先安装ffmpeg）：  
```ffmpeg -f v4l2 -s 240x320 -r 25 -vcodec mjpeg -i /dev/video0 -b:v 8000k -an output.avi```  
安装mplayer```apt install mpv```,用它打开刚才拍摄的内容```mpv output.avi```  
把它们结合起来，便实现了开发板实时采集热成像画面```ffmpeg -f v4l2 -s 320x240 -r 25 -vcodec mjpeg -i /dev/video0 -b:v 8000k -an -f avi pipe:1 | mpv -``` 
可以用这个命令查看设备支持的格式```v4l2-ctl --list-formats-ext --device=/dev/video0```,用这个命令调整画面参数，如帧率、大小和方向```ffmpeg -f v4l2 -s 320x240 -r 25 -vcodec mjpeg -i /dev/video0 -vf "scale=640:480,transpose=1" -b:v 8000k -an -f avi pipe:1 | mpv -```   

### 2.2 编写基于OpenCV的测试程序  
我们使用 V4L2 作为 OpenCV 的视频捕获后端来显示热成像画面。具体的C++程序与CMake已包含在test文件夹中，只需自行交叉编译成可执行文件即可。可以使用以下命令进行编译:在项目根目录下创建一个构建目录```mkdir build && cd build``` 运行CMake来配置项目并生成构建系统```cmake ..``` 编译项目```make```  
然而，这个程序仅实现了基础的画面显示，而且延迟较大，仍需进一步改进。  
实现效果如图：  
![a988fb31a7bfd4b0a868d47b9814ec9](https://github.com/LegionMay/InfraredThermalImagingDriverAssist/assets/110379545/73b2b35f-daaa-4281-9eeb-fc27b8a425dd)

### 2.3 将X-LINUX-AI包中的目标检测模型集成到程序中，初步实现目标检测功能  
首先参考[X-LINUX-AI_Developer_package](https://wiki.st.com/stm32mpu/wiki/X-LINUX-AI_Developer_package) 在PC上安装X-LINUX-AI开发包（包含我们所需的TensorFlow Lite库），安装成功后应当能在类似这个路径```~/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/include/tensorflow/lite```下找到我们所需的头文件，然后就可以着手编写实现目标检测的程序啦。  
我们使用了X-LINUX-AI包自带的```root@stm32mp1:/usr/local/demo-ai/computer-vision/models/coco_ssd_mobilenet#```路径下的目标检测模型```detect.tflite```进行测试，具体的CPP程序和CMakeLists文件包含在CV_Test路径下。  
交叉编译并推送到开发板执行后，我们成功实现了对热成像采集画面进行目标检测。然而，画面帧率极低，检测精度也不够理想。把程序改为两个线程后，帧率有一些提升，但仍无法满足需求。改进后的CPP程序及其CMakeLists文件位于MainTest路径。下一步，我们需要优化程序并利用开源数据集[CTIR_Dataset](https://gitee.com/bjtu_dx/ctir-dataset)训练适合我们自己的TF Lite模型。  
效果如图：  
![476c98716505c1f6fc781c66481836b](https://github.com/LegionMay/InfraredThermalImagingDriverAssist/assets/110379545/1e402bab-7e94-4f87-9ec9-513ed9e5e770)  
### 2.4 训练自己的目标检测模型  
在Ubuntu系统中安装了Anaconda之后，可以按照以下步骤来安装训练TensorFlow Lite模型所需的工具：
1. 打开终端：可以通过快捷键Ctrl + Alt + T打开一个新的终端窗口。
2. 创建新的虚拟环境：使用Anaconda创建一个新的虚拟环境，这样可以避免与系统中其他项目的依赖冲突。可以使用以下命令：  
   ```conda create -n tflite_env python=3.9```  
   这里python=3.9是指定Python的版本，tflite_env是新虚拟环境的名称。  
3.激活虚拟环境：创建虚拟环境后，使用以下命令来激活它：  
```conda activate tflite_env```  
4.安装TensorFlow Lite Model Maker：  
在激活的虚拟环境中，使用pip来安装TensorFlow Lite Model Maker：  
```pip install tflite-model-maker```  
5.验证安装：安装完成后，可以通过运行以下命令来验证TensorFlow Lite Model Maker是否正确安装：
```python -c "import tflite_model_maker"```
如果没有错误信息输出，那么安装就成功了。  
6.开始使用：现在可以开始使用TensorFlow Lite Model Maker来训练模型并将其转换为TensorFlow Lite格式了

  这里参考[这篇博客](https://blog.csdn.net/jiugeshao/article/details/124235916)编写了我们自己的模型训练脚本(位于Dataset路径下)，把开源数据集[CTIR_Dataset](https://gitee.com/bjtu_dx/ctir-dataset)分成train、test和valide三部分进行训练，

<img width="959" alt="52224db8d8fae946743add6ed805456" src="https://github.com/LegionMay/InfraredThermalImagingDriverAssist/assets/110379545/b5f43f2f-fe06-4e75-ad0a-26b333777112">


如此训练数十个小时后，我们得到了自己的第一个tflite目标检测模型。  

## 3 加速前进
### 3.1 GoogleEdgeTPU加速AI推理 
令人遗憾的是，STM32MP157的性能似乎无法满足我们模型推理的帧率要求，因此我们决定启用GoogleEdgeTPU对模型的推理进行加速。  
首先，在开发板上配置GoogleEdgeTPU 运行环境。参考(https://github.com/google-coral/edgetpu).  在开发环境上同样安装libedgetpu。
注意某些版本的tflite(>2.11)与libedgetpu版本可能不适配，程序执行时会在创建解释器时崩溃。我们这里使用的是v5.0.0版本的X-LINUX-AI包与libedgetpu.2.0。  
要使用EdgeTPU加速推理，还要使用EdgeTPUComplier把tflite模型转换为edgetpu支持的格式。  
这里调用edgetpu的api进行推理的相关代码段如下：  
```cpp
// 加载模型
    model = tflite::FlatBufferModel::BuildFromFile("/home/root/usr/NewTPU/Models/Mydetect_edgetpu.tflite");
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
```
### 3.2 使用RK3588加速推理（可选）  
如果你不喜欢EdgeTPU，可以选择使用RK3588加速推理，甚至直接改用RK3588.   
对于RKNN环境的配置和RKNN模型的转换，这里不做赘述，仅提供可以参考的python程序（RK3588路径下）。  
我们使用Socket与RK3588建立TCP连接，分块收发视频流，即可实现把AI推理过程转移到RK3588上。  
### 3.3 热融合显示功能  
热融合显示部分是同时捕获热成像画面和近红外（可见光）画面，利用OpenCV提供的图像加权融合函数实现的，具体代码如下：  
```cpp
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
```
### 3.4 GUI界面的绘制和触屏操作的处理  
GUI界面的绘制是利用OpenCV的绘图函数实现的，部分代码如下：  
```cpp
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
```
而触屏操作是通过OpenCV的鼠标点击回调函数实现的，相关代码如下：  
```cpp
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
```
### 3.5 TCP通信  
为了与车机建立高速而稳定的通信，我们利用Socket编程实现TCP网络通信，部分相关代码如下：  
```cpp
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
```
