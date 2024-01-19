import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
import cv2
import numpy as np

# 数据加载
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32') / 255
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32') / 255
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

# 创建序贯模型，并添加 dropout 层
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.25),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.25),
    Flatten(),
    Dense(120, activation='relu'),
    #Dropout(0.5),
    Dense(84, activation='relu'),
    #Dropout(0.5),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_x, train_y, batch_size=300, epochs=10, verbose=1, validation_data=(test_x, test_y))

# 保存模型
model.save('mnist_cnn_model.h5')

# 加载模型
model = load_model('mnist_cnn_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法加载图像: " + image_path)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return img


def extract_digits(img):
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 根据轮廓的横坐标排序
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

    digit_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        digit = img[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit / 255.0
        digit_images.append(digit)
    return digit_images


def predict_digits(digit_images):
    predictions = []
    for digit in digit_images:
        digit = np.expand_dims(digit, axis=0)
        digit = np.expand_dims(digit, axis=3)
        prediction = model.predict(digit)
        digit_predicted = np.argmax(prediction)
        predictions.append(digit_predicted)
    return predictions

# 预处理图像
img = preprocess_image('2021214584.jpg')  # 修改为您的图像路径

# 提取并预测数字
digit_images = extract_digits(img)
predictions = predict_digits(digit_images)

# 打印预测结果
print("预测的学号为:", predictions)









































































































































































