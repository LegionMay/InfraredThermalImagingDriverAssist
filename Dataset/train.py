import numpy as np
import os
import xml.etree.ElementTree as ET

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import cv2
from PIL import Image

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# 定义目标类别
classes = ['Pedestrian', 'Car', 'Truck', 'Bus', 'Cyclist']
# 为可视化定义颜色列表
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

# 定义训练函数
def train():
    # 标签字典，将类别映射到数字
    labels = {
        1: 'Pedestrian',
        2: 'Car',
        3: 'Truck',
        4: 'Bus',
        5: 'Cyclist'
    }
    # 训练集、验证集和测试集的路径
    train_imgs_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/train/JPEGImages"
    train_Anno_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/train/Annotations"
    valide_imgs_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/valide/JPEGImages"
    valide_Anno_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/valide/Annotations"
    test_imgs_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/test/JPEGImages"
    test_Anno_dir = "/home/osboxes/InfraredThermalImagingDriverAssist/CTIR_Dataset/test/Annotations"

    # 转换标注文件中的坐标值为整数
    convert_annotations_to_int(train_Anno_dir)
    convert_annotations_to_int(valide_Anno_dir)
    convert_annotations_to_int(test_Anno_dir)

    # 使用 Data Loader 从 Pascal VOC 格式的数据中加载训练数据、验证数据和测试数据
    traindata = object_detector.DataLoader.from_pascal_voc(train_imgs_dir, train_Anno_dir, labels)
    validata = object_detector.DataLoader.from_pascal_voc(valide_imgs_dir, valide_Anno_dir, labels)
    testdata = object_detector.DataLoader.from_pascal_voc(test_imgs_dir, test_Anno_dir, labels)

    # 模型规格
    spec = model_spec.get('ssd_mobilenet_v2')
    # 设置模型规格的 URI 和输入图片尺寸
    spec.input_image_shape = [320, 240]

    # 创建目标检测模型并进行训练
    model = object_detector.create(traindata, model_spec=spec, batch_size=6, train_whole_model=True,
                                    validation_data=validata, epochs=40)
    # 打印模型摘要信息
    model.summary()

    # 在测试数据上评估模型性能
    model.evaluate(testdata)
    # 导出训练好的模型
    model.export(export_dir='/home/osboxes/InfraredThermalImagingDriverAssist/Model/MyModel')

def convert_annotations_to_int(annotations_dir):
    for filename in os.listdir(annotations_dir):
        filepath = os.path.join(annotations_dir, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            bbox.find('xmin').text = str(xmin)
            bbox.find('ymin').text = str(ymin)
            bbox.find('xmax').text = str(xmax)
            bbox.find('ymax').text = str(ymax)
        tree.write(filepath)


# 图像预处理函数
def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image

# 目标检测函数
def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""

    # 获取模型的签名函数
    signature_fn = interpreter.get_signature_runner()
    # 将输入图像输入模型
    output = signature_fn(images=image)
    # 从模型获取所有输出
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

# 运行目标检测并绘制结果函数
def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # 加载模型所需的输入形状
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # 加载输入图像并对其进行预处理
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # 在输入图像上运行目标检测
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # 在输入图像上绘制检测结果
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
        # 将目标边界框从相对坐标转换为基于原始图像分辨率的绝对坐标
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        # 查找当前对象的类别索引
        class_id = int(obj['class_id'])

        # 在图像上绘制边界框和标签
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
        # 使标签对所有对象都可见
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 返回最终图像
    original_uint8 = original_image_np.astype(np.uint8)
    return original_uint8


if __name__ == '__main__':
    # 开始训练模型
    train()

    # 开始加载 TFLite 模型并预测图像
    DETECTION_THRESHOLD = 0.3
    model_path = '/home/osboxes/InfraredThermalImagingDriverAssist/Model/MyModel/model.tflite'
    TEMP_FILE = '/home/osboxes/InfraredThermalImagingDriverAssist/Model/MyModel/135.bmp'

    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 运行推理并在原始文件的本地副本上绘制检测结果
    detection_result_image = run_odt_and_draw_results(
        TEMP_FILE,
        interpreter,
        threshold=DETECTION_THRESHOLD
    )

    # 显示检测结果
    image = Image.fromarray(detection_result_image)
    image.show()
