import cv2
import numpy as np
import os
import datetime


# 将图片的BGR三通道值改变
def bek(img, i, j, k):
    img[i][j][0] = k  # B>220
    img[i][j][1] = k  # G>220
    img[i][j][2] = k  # R<20


# 导入图片，识别图片中的车道线，并判断车道线偏离
def line(img):
    global state  # 当前状态
    global cnt_num  # 计数器
    img1 = img.copy()
    height = img1.shape[0]  # 获取图片高度
    width = img1.shape[1]  # 获取图片宽度
    for i in range(img1.shape[0]):  # 将图片进行处理，将识别出车道线的蓝色部分全都以白色显示，其余部分都显示为黑色
        for j in range(img1.shape[1]):
            if img[i][j][0] > 220 and img[i][j][1] > 220 and img[i][j][2] < 30:
                bek(img1, i, j, 255)
            else:
                bek(img1, i, j, 0)

    # 膨胀 加强蓝色车道线部分
    kernel = np.ones((2, 2), np.uint8)
    dst = cv2.dilate(img1, kernel)
    # compose=np.hstack((img1,dst))  # 将原图与膨胀后的图片堆叠，查看区别
    # cv2.imshow("compose",compose)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 获得灰度图
    # 阈值的二值化操作，大于阈值使用255表示，小于阈值使用0表示
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 通过二值化的灰度图进行轮廓查找，找到车道线的轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    # 将轮廓进行降序排序，以便过滤部分识别有误情况，因为识别出的车道线一般面积较大，同时可以过滤部分重复识别的情况
    contours.sort(key=cnt_area, reverse=True)
    # 创建一个空的多维数组，用作保存处理后的图像
    img_res = np.empty(img.shape)
    # 两个flag用于判断此图中是否已经识别出两条斜率为正负的线，用于过滤某些干扰并节约时间
    flag1 = False
    flag2 = False
    # 存储两条直线的端点坐标，用于计算交点
    line1 = []
    line2 = []
    # 遍历所有轮廓
    for cnt in contours:
        # 如果两条直线都计算出来了，后面的轮廓不予以计算
        if flag1 is True and flag2 is True:
            break
        # 如果轮廓面积小于500，视作干扰，此时认为识别有误，即未识别出两条直线，可能只识别出一条直线，直接前往下一帧
        if cnt_area(cnt) < 500:
            break
        # 将轮廓用一个最小的矩阵包起来，获得矩阵的左上角点与宽高
        x, y, w, h = cv2.boundingRect(cnt)

        # 用绿色(0, 255, 0)来画出最小的矩形框架
        # cv2.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("test.jpg", test)
        # cv2.waitKey(0)

        # 使用fitLine函数自动拟合直线，大致获得斜率k的大小，仅用作判断，真实直线的具体参数后续还要调整
        [vx, vy, x4, y4] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        # 通过获得的自动拟合直线参数获得斜率
        k = vy / vx
        # 如果斜率过大，认为干扰，有时会将某些物体（如电线杆）误判
        if k > 10 or k < -10:
            continue
        # 判断此斜率的直线是否已存在，若存在，跳过此直线
        if (k > 0 and flag1 is True) or (k < 0 and flag2 is True):
            continue
        # 此拟合是将包裹轮廓的矩形的某条对角线作为拟合后的直线，用k判断是哪条对角线
        if k > 0:
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1, y1, x2, y2 = x, y + h, x + w, y
        '''测试代码 用于查看交点情况'''
        # if vx*a+vy*b<0:
        #     x1,y1,x2,y2=x2,y2,x1,y1
        # k = (y2 - y1) / (x2 - x1)
        # if(k<0):
        #    x2=int(x2+50)
        #    y2=int(y2+50*k)
        # else:
        #    x1 = int(x1- 50)
        #    y1 = int(y1 - 50 * k)

        # 绘制直线
        img_res = cv2.line(img_res, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 改变flag的状态，保存直线参数（两个端点）
        if k > 0:
            flag1 = True
            line1 = [x1, y1, x2, y2]
        else:
            flag2 = True
            line2 = [x1, y1, x2, y2]
    img_line = img.copy()
    # 若检测出两条直线，则求交点，若没检测出两条直线，则直接跳过
    if (len(line1) != 0) and (len(line2) != 0):
        # 定义临时变量，方便后续编码
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]
        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]
        # 求交点
        k1 = (y2 - y1) / (x2 - x1)
        k2 = (y4 - y3) / (x4 - x3)
        b1 = y1 - k1 * x1
        b2 = y3 - k2 * x3
        point_x = (b2 - b1) / (k1 - k2)
        point_y = k1 * point_x + b1
        # 获取当前帧内，两条直线交点的状态
        now_state = 0.4 * width < point_x < 0.6 * width and 0.4 * height < point_y < 0.6 * height
        # 如果当前状态与总体状态相同，重置计数器
        if now_state == state:
            cnt_num = 0
        # 如果不相同，计数器+1，直到连续6帧有此情况出现，就改变总体状态，为了防止某些帧的检测失误
        else:
            cnt_num += 1
            if cnt_num > 5:
                state = not state
    # 在图像上根据总体状态显现文本
    if state == False:
        cv2.putText(img_res, "WARNING:YOU ARE OFF TRACK!", (40, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_res, "IN LINE", (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 0, 255), 2)
    # 应用处理后的图片
    for i in range(img_res.shape[0]):
        for j in range(img_res.shape[1]):
            if img_res[i][j][0] != 0.0 or img_res[i][j][1] != 0.0 or img_res[
                    i][j][2] != 0.0:
                img_line[i][j] = img_res[i][j]
    return img_line


# 求轮廓面积
def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


# 获取视频
os.chdir("C:\\大三下\\大创\\计算机图像处理\\lane")
cap = cv2.VideoCapture('车道线_20210515T191543.avi')
# 获取宽度与高度
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# cv2.VideoWriter()指定写入视频帧编码格式
codec = cv2.VideoWriter_fourcc(*'DIVX')
# 指定存储文件夹
save_dir = os.path.join(os.getcwd(), "output")
# 指定输出文件名
file_name = "videofile_maksed_{:%Y%m%dT%H%M%S}.avi".format(
    datetime.datetime.now())
file_name = os.path.join(save_dir, file_name)
print(save_dir)
print(file_name)
# 输出视频流对象
output = cv2.VideoWriter(file_name, codec, 60.0, size)
# 初始化总体状态与计数器
state = True
cnt_num = 0
while cap.isOpened():
    # 读取帧图像
    ret, img = cap.read()
    # 处理
    img_line = line(img)
    cv2.imshow("res.jpg", img_line)
    output.write(img_line)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
