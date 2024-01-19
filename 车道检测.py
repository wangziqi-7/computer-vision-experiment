import cv2
import numpy as np

blur_ksize = 5
canny_lthreshold = 50
canny_hthreshold = 150
rho = 0.5
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20
def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        mask_color = (255,) * img.shape[2]
    else:
        mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
def process_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)

    # 感兴趣区域的顶点，根据图像大小自定义
    roi_vtx = np.array([[(-30, img.shape[0]), (img.shape[1] / 2 - 50, img.shape[0] / 2 + 50),
                         (img.shape[1] / 2 + 50, img.shape[0] / 2 + 50), (img.shape[1]+30, img.shape[0])]],
                       dtype=np.int32)
    roi_edges = roi_mask(edges, roi_vtx)
    # 使用霍夫变换检测直线
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # 将检测的线绘制在原图上
    res_img = cv2.addWeighted(img, 1, line_img, 0.8, 0)
    # 显示图像
    cv2.imshow('lane lines', res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 调用函数
process_image('test_image.jpg')
