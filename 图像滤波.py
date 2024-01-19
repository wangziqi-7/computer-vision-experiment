import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 加载图像并转换为灰度NumPy数组
def load_image(path):
    image_array=np.array(Image.open(path).convert('L'))
    print(image_array.shape)
    return image_array
def convolve(image, kernel):
    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    image_height = len(image)
    image_width = len(image[0])
    padded_image = [[0 for _ in range(image_width + 2 * pad_width)] for _ in range(image_height + 2 * pad_height)]
    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + pad_height][j + pad_width] = image[i][j]
    output = [[0 for _ in range(image_width)] for _ in range(image_height)]
    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            sum = 0
            for k in range(kernel_height):
                for l in range(kernel_width):
                    sum += kernel[k][l] * padded_image[i - pad_height + k][j - pad_width + l]
            output[i - pad_height][j - pad_width] = sum
    return output
# Sobel算子
def sobel_filter(image):
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    Gx = convolve(image, Kx)
    Gy = convolve(image, Ky)
    # 计算梯度幅值
    G = [[0 for _ in range(len(image[0]))] for _ in range(len(image))]
    for i in range(len(image)):
        for j in range(len(image[0])):
            G[i][j] = min(int((Gx[i][j]**2 + Gy[i][j]**2) ** 0.5), 255)
    return G
# 自定义卷积核
def custom_filter(image):
    kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    return convolve(image, kernel)
# 颜色直方图
def plot_histogram(image, histogram_path):
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    # 创建一个新的白色图像来绘制直方图
    histogram_image = Image.new('RGB', (256, 100), (255, 255, 255))
    draw = ImageDraw.Draw(histogram_image)
    # 计算直方图的最大频次用于标准化
    max_freq = max(histogram)
    for i in range(256):
        # 标准化频次
        freq = int((histogram[i] / max_freq) * 100)
        # 绘制直方图的条形（竖线）
        draw.line((i, 100, i, 100 - freq), fill='black')
    # 保存直方图图像到相对路径
    histogram_image.save(histogram_path)
# 纹理特征提取
def extract_texture_features(image):
    total = 0
    for row in image:
        for pixel in row:
            total += pixel
    mean = total / (len(image) * len(image[0]))
    variance_sum = 0
    for row in image:
        for pixel in row:
            variance_sum += (pixel - mean) ** 2
    variance = variance_sum / (len(image) * len(image[0]))
    std_dev = variance ** 0.5
    return std_dev
# 加载图像
image_path = 'xiangrikui.jpg'
image = load_image(image_path)
# 应用Sobel滤波器
sobel_image = sobel_filter(image)
# 应用自定义卷积核
custom_filtered_image = custom_filter(image)
plot_histogram(image, 'image_histogram.jpg')
# 提取纹理特征
texture_features = extract_texture_features(image)
# 保存纹理特征
np.save('texture_features.npy', texture_features)
# 转换处理后的图像为 NumPy 数组
sobel_image_array = np.array(sobel_image, dtype=np.uint8)
custom_filtered_image_array = np.array(custom_filtered_image, dtype=np.uint8)
# 保存处理后的图像的相对路径
output_sobel_path = 'sobel_filtered_image.jpg'
output_custom_path = 'custom_filtered_image.jpg'
# 保存处理后的图像
Image.fromarray(sobel_image_array).save(output_sobel_path)
Image.fromarray(custom_filtered_image_array).save(output_custom_path)

plt.imshow(sobel_image_array, cmap='gray')
plt.title('Sobel Filtered Image')
plt.show()
plt.imshow(custom_filtered_image_array, cmap='gray')
plt.title('Custom Filtered Image')
plt.show()
