cmake_minimum_required(VERSION 3.0)
project(MainTest)

set(CMAKE_BUILD_TYPE Debug)


# 设置交叉编译器路径
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 指定交叉编译器（使用完整路径）
set(CMAKE_C_COMPILER /home/osboxes/Workspace/build-root-2023.02.2/buildroot/output/host/bin/arm-linux-gcc)
set(CMAKE_CXX_COMPILER /home/osboxes/Workspace/build-root-2023.02.2/buildroot/output/host/bin/arm-linux-g++)

# 设置OpenCV路径
set(OpenCV_DIR "/home/osboxes/opencv/opencv_install/lib/cmake/opencv4")

# 查找OpenCV库
find_package(OpenCV REQUIRED core imgproc highgui videoio)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置TensorFlow Lite路径
set(TFLITE_PARENT_DIR "/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/include")
include_directories(${TFLITE_PARENT_DIR})

# 添加源文件
add_executable(MainTest "main.cpp")

# 链接OpenCV库
target_link_libraries(MainTest ${OpenCV_LIBS})

# 链接TensorFlow Lite库
set(TFLITE_LIB_DIR "/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib")
link_directories(${TFLITE_LIB_DIR})
target_link_libraries(MainTest "/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib/libtensorflow-lite.so")


# 设置C++11支持
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

