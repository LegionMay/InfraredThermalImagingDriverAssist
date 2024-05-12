cmake_minimum_required(VERSION 3.0)
project(Go)

# 设置交叉编译器路径
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 指定交叉编译器（使用完整路径）
set(CMAKE_C_COMPILER /home/osboxes/Workspace/build-root-2023.02.2/buildroot/output/host/bin/arm-linux-gcc)
set(CMAKE_CXX_COMPILER /home/osboxes/Workspace/build-root-2023.02.2/buildroot/output/host/bin/arm-linux-g++)

# 设置OpenCV路径
set(OpenCV_DIR "/home/osboxes/opencv/opencv_install/lib/cmake/opencv4")

# 查找OpenCV库
find_package(OpenCV REQUIRED core imgproc highgui videoio) # 只列出你需要的模块

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加源文件
add_executable(Go "Goo.cpp") # 更改为您的源文件名

# 链接OpenCV库，只链接你需要的模块
target_link_libraries(Go ${OpenCV_LIBS}) # 使用变量链接所有找到的库

# 设置C++11支持
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)