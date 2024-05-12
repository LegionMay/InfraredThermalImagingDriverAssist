# InfraredThermalImagingDriverAssist  
## 前言  
随着汽车行业的快速发展，驾驶安全已成为公众关注的焦点。据交通安全部门统计，低能见度环境下的交通事故率明显高于正常天气条件。传统的驾驶辅助系统依赖于可见光摄像头，但在雾天、夜间等低能见度环境下，往往难以准确识别道路障碍物，增加了交通事故的风险。为了解决这一问题，我们开发了一款基于STM32MP157F-DK2的红外热成像智能车载驾驶辅助系统，通过在该主控平台上部署Linux操作系统，结合可见光摄像头和红外热成像模块，并采用图像处理算法和嵌入式人工智能技术，实现在低能见度环境下对障碍物、人员、车辆的智能识别和碰撞预警，从而提高驾驶安全性。该系统能够有效解决低能见度环境下的驾驶安全问题，减少因视线不良导致的交通事故，提升夜间或恶劣天气条件下的驾驶体验，同时为智能驾驶辅助系统的发展带来了新的机遇。  
## 1. 迈出第一步
### 1.1 配置开发环境，点亮LCD  
这里选择直接在Ubuntu上进行开发，为方便文件的传输，使用```sudo vmhgfs-fuse .host:/<共享文件夹名> /mnt/hgfs -o subtype=vmhgfs-fuse,allow_other```命令将一个共享文件夹挂载在Ubuntu的/mnt/hgfs目录。  
开发环境的配置通常是一个繁琐的过程，这里参考了ST官方的教程 [STM32MP157x-DK2](https://wiki.stmicroelectronics.cn/stm32mpu/wiki/Getting_started/STM32MP1_boards/STM32MP157x-DK2/Develop_on_Arm%C2%AE_Cortex%C2%AE-A7 )，一步一步跟着来。 
同时直接使用了参加st线下buidroot活动时获得的其官方配置好的虚拟机，Ubuntu版本为22.04。  
值得注意的是使用CubeProgrammer烧录时自动生成的二进制文件存储路径可能不准确，而且手动配置时会自动添加后缀。另外STM32MP157F-DK2实测供电至少为5V 2A才能点亮屏幕。  
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

## 2 迈出第二步
### 2.1 为开发板安装X-LINUX-AI软件包并测试热成像模块    
参考[X-LINUX-AI 入门包](https://wiki.st.com/stm32mpu/wiki/X-LINUX-AI_Starter_package)安装X-LINUX-AI软件包，其中包括了我们所需的opencv和tensorflow lite。  
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




