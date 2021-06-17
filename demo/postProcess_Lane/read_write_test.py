import cv2
import os
import datetime
os.chdir("C:\\Users\\zjx61\\Desktop")
cap = cv2.VideoCapture('jsrecord_2021-06-13-09-38-55.mp4')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
codec = cv2.VideoWriter_fourcc(*'DIVX')
save_dir = os.path.join(os.getcwd(), "output")
file_name = "videofile_maksed_{:%Y%m%dT%H%M%S}.avi".format(
    datetime.datetime.now())
file_name = os.path.join(save_dir, file_name)
print(save_dir)
print(file_name)
output = cv2.VideoWriter(file_name, codec, 240.0, size)
while cap.isOpened():
    ret, img = cap.read()
    # 处理
    img_line = img
    cv2.imshow("res.jpg", img_line)
    output.write(img_line)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()