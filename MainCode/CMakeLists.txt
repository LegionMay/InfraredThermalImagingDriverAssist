
cmake_minimum_required(VERSION 3.0)
project(TempMP157Go2)

# 设置C++11支持
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


set(CMAKE_BUILD_TYPE Debug)

# 设置交叉编译器路径
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# 指定交叉编译器
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

# 设置Edge TPU路径
set(EDGETPU_PARENT_DIR "/home/osboxes/EdgeTPU/edgetpu/libedgetpu/")  
include_directories(${EDGETPU_PARENT_DIR})

# 添加源文件
add_executable(${PROJECT_NAME} "main.cpp")

# 链接OpenCV库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# 链接TensorFlow Lite库
set(TFLITE_LIB_DIR "/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib")
link_directories(${TFLITE_LIB_DIR})
target_link_libraries(${PROJECT_NAME} ${TFLITE_LIB_DIR}/libtensorflow-lite.so)

### 链接libusb库
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib")

link_directories(/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/lib)
link_directories(/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib)

target_link_libraries(${PROJECT_NAME}
    /home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/lib/libusb-1.0.so.0
    /home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib/libflatbuffers.so.23
)

###

# 链接Edge TPU库

set(EDGETPU_LIB_DIR "/home/osboxes/EdgeTPU/edgetpu/libedgetpu/throttled/armv7a")
#set(EDGETPU_LIB_DIR "/home/osboxes/Workspace/STM32MPU-Ecosystem-v5.0.2/Developer-Package/SDK/sysroots/cortexa7t2hf-neon-vfpv4-ostl-linux-gnueabi/usr/lib")

link_directories(${EDGETPU_LIB_DIR})
target_link_libraries(${PROJECT_NAME} ${EDGETPU_LIB_DIR}/libedgetpu.so.1.0)


